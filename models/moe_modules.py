#-------------------------------------------------------------------------------
# Name:        moe_modules.py
# Purpose:     reusable Mixture-of-Experts layers for RigNet models
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm, Module, ModuleList


def _expert_block(in_channels, out_channels, use_norm=True):
    layers = [Linear(in_channels, out_channels), ReLU()]
    if use_norm:
        layers.append(LayerNorm(out_channels))
    return Sequential(*layers)


def switch_load_balance_loss(router_probs, top_k_indices, num_experts):
    """Switch Transformer / GShard auxiliary load-balancing loss."""
    if router_probs.numel() == 0:
        return router_probs.new_zeros(())
    density = router_probs.mean(dim=0)
    routing = torch.zeros_like(router_probs)
    routing.scatter_(1, top_k_indices, 1.0)
    density_proxy = routing.mean(dim=0)
    return num_experts * torch.sum(density_proxy * density)


class MoELayer(Module):
    """Top-k sparse Mixture-of-Experts layer with optional separate routing features."""

    def __init__(self, in_channels, out_channels, num_experts=4, top_k=2, use_norm=True, gate_input_dim=None):
        super(MoELayer, self).__init__()
        if num_experts < 1:
            raise ValueError('num_experts must be >= 1')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate_input_dim = gate_input_dim if gate_input_dim is not None else in_channels
        self.gate = Linear(self.gate_input_dim, num_experts)
        self.experts = ModuleList([
            _expert_block(in_channels, out_channels, use_norm=use_norm)
            for _ in range(num_experts)
        ])
        self.aux_loss = None

    def forward(self, x, gate_input=None):
        gate_features = gate_input if gate_input is not None else x
        router_logits = self.gate(gate_features)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        self.aux_loss = switch_load_balance_loss(router_probs, top_k_indices, self.num_experts)

        out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)
        for expert_idx, expert in enumerate(self.experts):
            for k in range(self.top_k):
                mask = top_k_indices[:, k] == expert_idx
                if mask.any():
                    expert_out = expert(x[mask])
                    out[mask] += top_k_probs[mask, k:k + 1] * expert_out
        return out


class MoEMLP(Module):
    """Multi-layer MoE stack that replaces MLP(channels) in RigNet heads."""

    def __init__(self, channels, num_experts=4, top_k=2, use_norm=True, gate_input_dim=None):
        super(MoEMLP, self).__init__()
        if len(channels) < 2:
            raise ValueError('channels must contain at least input and output dimensions')
        self.gate_input_dim = gate_input_dim
        self.layers = ModuleList([
            MoELayer(
                channels[i - 1],
                channels[i],
                num_experts=num_experts,
                top_k=top_k,
                use_norm=use_norm,
                gate_input_dim=gate_input_dim if i == 1 else None,
            )
            for i in range(1, len(channels))
        ])
        self.aux_loss = None

    def forward(self, x, gate_input=None):
        h = x
        aux_loss = x.new_zeros(())
        for layer_idx, layer in enumerate(self.layers):
            routing = gate_input if layer_idx == 0 else None
            h = layer(h, gate_input=routing)
            aux_loss = aux_loss + layer.aux_loss
        self.aux_loss = aux_loss
        return h


def collect_moe_load_balance_loss(module):
    """Sum auxiliary load-balancing losses from all MoE blocks in a module tree."""
    total = None
    for submodule in module.modules():
        if isinstance(submodule, (MoELayer, MoEMLP)) and submodule.aux_loss is not None:
            total = submodule.aux_loss if total is None else total + submodule.aux_loss
    if total is None:
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        return torch.tensor(0.0, device=device)
    return total
