#-------------------------------------------------------------------------------
# Name:        GCN.py
# Purpose:     definition of joint prediction module.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import torch
from models.gcn_basic_modules import MLP, GCU
from models.iterative_mask_refinement import IterativeMaskRefinement
from torch_scatter import scatter_max, scatter_mean
from torch.nn import Sequential, Dropout, Linear, ReLU, Parameter

H_DIM = 64 + 256 + 512  # persistent per-vertex features from GCU stack


class JointPredNet(torch.nn.Module):
    def __init__(self, out_channels, input_normal, arch, aggr='max'):
        super(JointPredNet, self).__init__()
        self.input_normal = input_normal
        self.arch = arch
        if self.input_normal:
            self.input_channel = 6
        else:
            self.input_channel = 3
        self.gcu_1 = GCU(in_channels=self.input_channel, out_channels=64, aggr=aggr)
        self.gcu_2 = GCU(in_channels=64, out_channels=256, aggr=aggr)
        self.gcu_3 = GCU(in_channels=256, out_channels=512, aggr=aggr)
        # feature compression
        self.mlp_glb = MLP([(64 + 256 + 512), 1024])
        self.mlp_tramsform = Sequential(MLP([1024 + self.input_channel + 64 + 256 +512, 1024, 256]),
                                        Dropout(0.7), Linear(256, out_channels))
        if self.arch == 'jointnet':
            torch.nn.init.zeros_(self.mlp_tramsform[2].weight)
            torch.nn.init.zeros_(self.mlp_tramsform[2].bias)

    def _encode(self, data):
        if self.input_normal:
            x = torch.cat([data.pos, data.x], dim=1)
        else:
            x = data.pos
        geo_edge_index, tpl_edge_index = data.geo_edge_index, data.tpl_edge_index

        x_1 = self.gcu_1(x, tpl_edge_index, geo_edge_index)
        x_2 = self.gcu_2(x_1, tpl_edge_index, geo_edge_index)
        x_3 = self.gcu_3(x_2, tpl_edge_index, geo_edge_index)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))

        x_global, _ = scatter_max(x_4, data.batch, dim=0)
        x_global = torch.repeat_interleave(x_global, torch.bincount(data.batch), dim=0)

        h = torch.cat([x_1, x_2, x_3], dim=1)
        return {
            "h": h,
            "x": x,
            "x_1": x_1,
            "x_2": x_2,
            "x_3": x_3,
            "x_global": x_global,
        }

    def encode(self, data):
        """Return persistent per-vertex GCU features and context."""
        return self._encode(data)

    def decode(self, encoded):
        """Map encoded features to output logits/displacements."""
        x_5 = torch.cat(
            [encoded["x_global"], encoded["x"], encoded["x_1"], encoded["x_2"], encoded["x_3"]],
            dim=1,
        )
        out = self.mlp_tramsform(x_5)
        if self.arch == 'jointnet':
            out = torch.tanh(out)
        return out

    def forward(self, data):
        encoded = self._encode(data)
        return self.decode(encoded)


class JOINTNET_MASKNET_MEANSHIFT(torch.nn.Module):
    """
    JointNet (once) -> MaskNet initial logits -> IterativeMaskRefinement -> mean-shift attention.

    Pipeline per forward pass:
      1. JointNet predicts displacement (fixed for all refinement steps).
      2. MaskNet backbone produces h and initial mask logits.
      3. IterativeMaskRefinement updates logits via residual deltas.
      4. Final sigmoid(attention) is consumed by the existing mean-shift routine.
    """

    def __init__(
        self,
        num_refine_steps=2,
        num_experts=1,
        shared_refinement=True,
        refine_hidden_dim=256,
        gating_hidden_dim=128,
    ):
        super(JOINTNET_MASKNET_MEANSHIFT, self).__init__()
        self.jointnet = JointPredNet(3, input_normal=False, arch='jointnet', aggr='max')
        self.masknet = JointPredNet(1, input_normal=False, arch='masknet', aggr='max')
        self.bandwidth = Parameter(torch.Tensor(1))
        self.bandwidth.data.fill_(0.04)

        self.num_refine_steps = num_refine_steps
        self.num_experts = num_experts
        self.shared_refinement = shared_refinement

        self.refinement = IterativeMaskRefinement(
            h_dim=H_DIM,
            num_refine_steps=num_refine_steps,
            num_experts=num_experts,
            shared_refinement=shared_refinement,
            refine_hidden_dim=refine_hidden_dim,
            gating_hidden_dim=gating_hidden_dim,
        )
        self._jointnet_frozen = False
        self._masknet_frozen = False

    def freeze_jointnet(self):
        """Freeze JointNet: no gradients, eval mode, detached forward."""
        for param in self.jointnet.parameters():
            param.requires_grad = False
        self.jointnet.eval()
        self._jointnet_frozen = True

    def verify_jointnet_frozen(self):
        """Sanity check that JointNet cannot receive optimizer updates."""
        assert self._jointnet_frozen, "JointNet must be frozen for iterative refinement experiment"
        for name, param in self.jointnet.named_parameters():
            assert not param.requires_grad, f"JointNet param {name} still requires grad"
        assert not any(p.requires_grad for p in self.jointnet.parameters())

    def freeze_masknet(self):
        """Freeze MaskNet: refinement then sees deterministic, stationary h and logits_0."""
        for param in self.masknet.parameters():
            param.requires_grad = False
        self.masknet.eval()
        self._masknet_frozen = True

    def verify_masknet_frozen(self):
        assert self._masknet_frozen, "freeze_masknet() was not called"
        for name, param in self.masknet.named_parameters():
            assert not param.requires_grad, f"MaskNet param {name} still requires grad"

    def train(self, mode=True):
        """Keep frozen submodules in eval mode so dropout/BatchNorm stay deterministic."""
        super(JOINTNET_MASKNET_MEANSHIFT, self).train(mode)
        if self._jointnet_frozen:
            self.jointnet.eval()
        if self._masknet_frozen:
            self.masknet.eval()
        return self

    def forward(self, data, return_refinement=False):
        if self._jointnet_frozen:
            with torch.no_grad():
                x_offset = self.jointnet(data)
            x_offset = x_offset.detach()
        else:
            x_offset = self.jointnet(data)

        if self._masknet_frozen:
            with torch.no_grad():
                encoded = self.masknet.encode(data)
                logits_0 = self.masknet.decode(encoded)
            h = encoded["h"].detach()
            logits_0 = logits_0.detach()
        else:
            encoded = self.masknet.encode(data)
            h = encoded["h"]
            logits_0 = self.masknet.decode(encoded)

        refine_out = self.refinement(h, data.pos, x_offset, logits_0, data.batch)
        mask_logits = refine_out["logits_steps"][-1]
        mask_prob = torch.sigmoid(mask_logits)
        q_final = refine_out["q_steps"][-1]

        if return_refinement:
            return {
                "x_offset": x_offset,
                "mask_logits": mask_logits,
                "mask_prob": mask_prob,
                "bandwidth": self.bandwidth,
                "logits_steps": refine_out["logits_steps"],
                "attention_steps": refine_out["attention_steps"],
                "displacement_steps": refine_out["displacement_steps"],
                "q_steps": refine_out["q_steps"],
                "delta_logits_steps": refine_out["delta_logits_steps"],
                "gate_entropy_steps": refine_out["gate_entropy_steps"],
                "prob_shift_steps": refine_out["prob_shift_steps"],
                "alpha_steps": refine_out["alpha_steps"],
                "h": h,
                "q_final": q_final,
            }

        return x_offset, mask_logits, mask_prob, self.bandwidth
