#-------------------------------------------------------------------------------
# Name:        GCN.py
# Purpose:     definition of joint prediction module.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import torch
from models.gcn_basic_modules import MLP, GCU
from torch_scatter import scatter_max, scatter_mean
from torch.nn import Sequential, Dropout, Linear, ReLU, Parameter, LayerNorm, MultiheadAttention


class JointPredNet(torch.nn.Module):
    def __init__(self, out_channels, input_normal, arch, aggr='max', use_mha=False, mha_heads=2):
        super(JointPredNet, self).__init__()
        self.input_normal = input_normal
        self.arch = arch
        self.use_mha = use_mha
        if self.input_normal:
            self.input_channel = 6
        else:
            self.input_channel = 3
        self.gcu_1 = GCU(in_channels=self.input_channel, out_channels=64, aggr=aggr)
        self.gcu_2 = GCU(in_channels=64, out_channels=256, aggr=aggr)
        self.gcu_3 = GCU(in_channels=256, out_channels=512, aggr=aggr)
        if self.use_mha:
            self.mha = MultiheadAttention(
                embed_dim=512,
                num_heads=mha_heads,
                dropout=0.1,
                batch_first=True,
            )
            self.mha_norm = LayerNorm(512)
        # feature compression
        self.mlp_glb = MLP([(64 + 256 + 512), 1024])
        self.mlp_tramsform = Sequential(MLP([1024 + self.input_channel + 64 + 256 +512, 1024, 256]),
                                        Dropout(0.7), Linear(256, out_channels))
        if self.arch == 'jointnet':
            torch.nn.init.zeros_(self.mlp_tramsform[2].weight)
            torch.nn.init.zeros_(self.mlp_tramsform[2].bias)

    def _apply_mha(self, x_3, batch):
        """Self-attention per graph; vertices from different meshes never attend to each other."""
        unique_batches = batch.unique(sorted=True)
        if len(unique_batches) == 1:
            x_seq = x_3.unsqueeze(0)
            x_att, _ = self.mha(x_seq, x_seq, x_seq)
            return self.mha_norm(x_seq + x_att).squeeze(0)

        out = torch.empty_like(x_3)
        for graph_id in unique_batches:
            node_mask = batch == graph_id
            x_i = x_3[node_mask].unsqueeze(0)
            x_att, _ = self.mha(x_i, x_i, x_i)
            out[node_mask] = self.mha_norm(x_i + x_att).squeeze(0)
        return out

    def forward(self, data):
        if self.input_normal:
            x = torch.cat([data.pos, data.x], dim=1)
        else:
            x = data.pos
        geo_edge_index, tpl_edge_index, batch = data.geo_edge_index, data.tpl_edge_index, data.batch

        x_1 = self.gcu_1(x, tpl_edge_index, geo_edge_index)
        x_2 = self.gcu_2(x_1, tpl_edge_index, geo_edge_index)
        x_3 = self.gcu_3(x_2, tpl_edge_index, geo_edge_index)
        if self.use_mha:
            x_3 = self._apply_mha(x_3, batch)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))

        x_global, _ = scatter_max(x_4, data.batch, dim=0)
        #x_global_mean = scatter_mean(x_4, data.batch, dim=0)
        #x_global = torch.cat([x_global_max, x_global_mean], dim=1)
        x_global = torch.repeat_interleave(x_global, torch.bincount(data.batch), dim=0)

        x_5 = torch.cat([x_global, x, x_1, x_2, x_3], dim=1)
        out = self.mlp_tramsform(x_5)
        if self.arch == 'jointnet':
            out = torch.tanh(out)
        return out


class JOINTNET_MASKNET_MEANSHIFT(torch.nn.Module):
    def __init__(self, mask_mha=False, mha_heads=2):
        super(JOINTNET_MASKNET_MEANSHIFT, self).__init__()
        self.jointnet = JointPredNet(3, input_normal=False, arch='jointnet', aggr='max')
        self.masknet = JointPredNet(
            1, input_normal=False, arch='masknet', aggr='max',
            use_mha=mask_mha, mha_heads=mha_heads,
        )
        self.bandwidth = Parameter(torch.Tensor(1))
        self.bandwidth.data.fill_(0.04)

    def forward(self, data):
        x_offset = self.jointnet(data)
        x_mask_prob_0 = self.masknet(data)
        x_mask_prob = torch.sigmoid(x_mask_prob_0)
        return x_offset, x_mask_prob_0, x_mask_prob, self.bandwidth
