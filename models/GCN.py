#-------------------------------------------------------------------------------
# Name:        GCN.py
# Purpose:     definition of joint prediction module.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import torch
from models.gcn_basic_modules import MLP, GCU
from models.moe_modules import (
    MoEMLP, MOE_NUM_EXPERTS, MOE_TOP_K, MOE_ROUTER_NOISE, MOE_ROUTER_TEMPERATURE,
)
from torch_scatter import scatter_max, scatter_mean
from torch.nn import Sequential, Dropout, Linear, Parameter


class JointPredNet(torch.nn.Module):
    def __init__(self, out_channels, input_normal, arch, aggr='max', use_moe=True,
                 num_experts=MOE_NUM_EXPERTS, top_k=MOE_TOP_K, router_noise=MOE_ROUTER_NOISE,
                 router_temperature=MOE_ROUTER_TEMPERATURE):
        super(JointPredNet, self).__init__()
        self.input_normal = input_normal
        self.arch = arch
        self.use_moe = use_moe
        if self.input_normal:
            self.input_channel = 6
        else:
            self.input_channel = 3
        self.gcu_1 = GCU(in_channels=self.input_channel, out_channels=64, aggr=aggr)
        self.gcu_2 = GCU(in_channels=64, out_channels=256, aggr=aggr)
        self.gcu_3 = GCU(in_channels=256, out_channels=512, aggr=aggr)

        skip_dim = 64 + 256 + 512
        transform_input_dim = 1024 + self.input_channel + skip_dim
        transform_gate_dim = self.input_channel + 64
        if self.use_moe:
            self.mlp_glb = MoEMLP(
                [skip_dim, 1024],
                num_experts=num_experts,
                top_k=top_k,
                gate_input_dim=self.input_channel + skip_dim,
                router_noise=router_noise,
                router_temperature=router_temperature,
            )
            self.mlp_transform_body = MoEMLP(
                [transform_input_dim, 1024, 256],
                num_experts=num_experts,
                top_k=top_k,
                gate_input_dim=transform_gate_dim,
                router_noise=router_noise,
                router_temperature=router_temperature,
            )
            self.mlp_tramsform = Sequential(
                self.mlp_transform_body,
                Dropout(0.7),
                Linear(256, out_channels),
            )
        else:
            self.mlp_glb = MLP([skip_dim, 1024])
            self.mlp_tramsform = Sequential(
                MLP([transform_input_dim, 1024, 256]),
                Dropout(0.7),
                Linear(256, out_channels),
            )

        if self.arch == 'jointnet':
            torch.nn.init.zeros_(self.mlp_tramsform[-1].weight)
            torch.nn.init.zeros_(self.mlp_tramsform[-1].bias)

    def forward(self, data):
        if self.input_normal:
            x = torch.cat([data.pos, data.x], dim=1)
        else:
            x = data.pos
        geo_edge_index, tpl_edge_index, batch = data.geo_edge_index, data.tpl_edge_index, data.batch

        x_1 = self.gcu_1(x, tpl_edge_index, geo_edge_index)
        x_2 = self.gcu_2(x_1, tpl_edge_index, geo_edge_index)
        x_3 = self.gcu_3(x_2, tpl_edge_index, geo_edge_index)
        skip_features = torch.cat([x_1, x_2, x_3], dim=1)
        if self.use_moe:
            x_4 = self.mlp_glb(skip_features, gate_input=torch.cat([x, skip_features], dim=1))
        else:
            x_4 = self.mlp_glb(skip_features)

        x_global, _ = scatter_max(x_4, data.batch, dim=0)
        #x_global_mean = scatter_mean(x_4, data.batch, dim=0)
        #x_global = torch.cat([x_global_max, x_global_mean], dim=1)
        x_global = torch.repeat_interleave(x_global, torch.bincount(data.batch), dim=0)

        x_5 = torch.cat([x_global, x, x_1, x_2, x_3], dim=1)
        if self.use_moe:
            hidden = self.mlp_transform_body(x_5, gate_input=torch.cat([x, x_1], dim=1))
            out = self.mlp_tramsform[-1](self.mlp_tramsform[1](hidden))
        else:
            out = self.mlp_tramsform(x_5)
        if self.arch == 'jointnet':
            out = torch.tanh(out)
        return out


class JOINTNET_MASKNET_MEANSHIFT(torch.nn.Module):
    def __init__(self, use_moe=True, use_moe_jointnet=None, use_moe_masknet=None,
                 num_experts=MOE_NUM_EXPERTS, top_k=MOE_TOP_K, router_noise=MOE_ROUTER_NOISE,
                 router_temperature=MOE_ROUTER_TEMPERATURE):
        super(JOINTNET_MASKNET_MEANSHIFT, self).__init__()
        if use_moe_jointnet is None:
            use_moe_jointnet = use_moe
        if use_moe_masknet is None:
            use_moe_masknet = use_moe
        self.jointnet = JointPredNet(
            3, input_normal=False, arch='jointnet', aggr='max',
            use_moe=use_moe_jointnet, num_experts=num_experts, top_k=top_k,
            router_noise=router_noise, router_temperature=router_temperature,
        )
        self.masknet = JointPredNet(
            1, input_normal=False, arch='masknet', aggr='max',
            use_moe=use_moe_masknet, num_experts=num_experts, top_k=top_k,
            router_noise=router_noise, router_temperature=router_temperature,
        )
        self.bandwidth = Parameter(torch.Tensor(1))
        self.bandwidth.data.fill_(0.04)

    def forward(self, data):
        x_offset = self.jointnet(data)
        x_mask_prob_0 = self.masknet(data)
        x_mask_prob = torch.sigmoid(x_mask_prob_0)
        return x_offset, x_mask_prob_0, x_mask_prob, self.bandwidth
