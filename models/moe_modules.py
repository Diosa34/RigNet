#-------------------------------------------------------------------------------
# Name:        moe_modules.py
# Purpose:     reusable Mixture-of-Experts layers for RigNet models
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm, Module, ModuleList


# Tuned defaults: top-1 routing + deeper gate encourages geometric specialization.
MOE_NUM_EXPERTS = 6
MOE_TOP_K = 1
MOE_ROUTER_NOISE = 0.01
MOE_GATE_HIDDEN_RATIO = 4

# BoneNet has a single MoE stack on pair fusion; more experts help cover pair modes.
MOE_NUM_EXPERTS_BONENET = 8
MOE_TOP_K_BONENET = 1


def _expert_block(in_channels, out_channels, use_norm=True):
    layers = [Linear(in_channels, out_channels), ReLU()]
    if use_norm:
        layers.append(LayerNorm(out_channels))
    return Sequential(*layers)


class MoEGate(Module):
    """Small MLP gate for sharper, geometry-aware expert routing."""

    def __init__(self, in_channels, num_experts, hidden_ratio=MOE_GATE_HIDDEN_RATIO):
        super(MoEGate, self).__init__()
        hidden_dim = max(in_channels // hidden_ratio, num_experts * 2, 16)
        self.net = Sequential(
            Linear(in_channels, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_experts),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.net:
            if isinstance(module, Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
                torch.nn.init.zeros_(module.bias)

    def forward(self, gate_features):
        return self.net(gate_features)


def switch_load_balance_loss(router_probs, top_k_indices, num_experts, top_k):
    """Switch Transformer / GShard auxiliary load-balancing loss."""
    if router_probs.numel() == 0:
        return router_probs.new_zeros(())
    density = router_probs.mean(dim=0)
    routing = torch.zeros_like(router_probs)
    routing.scatter_(1, top_k_indices, 1.0)
    density_proxy = routing.mean(dim=0) / max(float(top_k), 1.0)
    return num_experts * torch.sum(density_proxy * density)


class MoELayer(Module):
    """Top-k sparse Mixture-of-Experts layer with MLP gate and optional routing features."""

    def __init__(self, in_channels, out_channels, num_experts=MOE_NUM_EXPERTS, top_k=MOE_TOP_K,
                 use_norm=True, gate_input_dim=None, router_noise=MOE_ROUTER_NOISE):
        super(MoELayer, self).__init__()
        if num_experts < 1:
            raise ValueError('num_experts must be >= 1')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router_noise = router_noise
        self.gate_input_dim = gate_input_dim if gate_input_dim is not None else in_channels
        self.gate = MoEGate(self.gate_input_dim, num_experts)
        self.experts = ModuleList([
            _expert_block(in_channels, out_channels, use_norm=use_norm)
            for _ in range(num_experts)
        ])
        self.aux_loss = None
        self.expert_usage = None
        self.router_entropy = None

    def forward(self, x, gate_input=None):
        gate_features = gate_input if gate_input is not None else x
        router_logits = self.gate(gate_features)
        if self.training and self.router_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        routing = torch.zeros_like(router_probs)
        routing.scatter_(1, top_k_indices, 1.0)
        self.expert_usage = (routing.mean(dim=0) / max(float(self.top_k), 1.0)).detach()
        self.router_entropy = (
            -(router_probs * (router_probs + 1e-9).log()).sum(dim=-1).mean().detach()
        )
        self.aux_loss = switch_load_balance_loss(
            router_probs, top_k_indices, self.num_experts, self.top_k,
        )

        out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)
        for expert_idx, expert in enumerate(self.experts):
            for k in range(self.top_k):
                mask = top_k_indices[:, k] == expert_idx
                if mask.any():
                    expert_out = expert(x[mask])
                    out[mask] = out[mask] + top_k_probs[mask, k:k + 1] * expert_out
        return out


class MoEMLP(Module):
    """Multi-layer MoE stack that replaces MLP(channels) in RigNet heads."""

    def __init__(self, channels, num_experts=MOE_NUM_EXPERTS, top_k=MOE_TOP_K, use_norm=True,
                 gate_input_dim=None, router_noise=MOE_ROUTER_NOISE, route_deep_layers=True):
        super(MoEMLP, self).__init__()
        if len(channels) < 2:
            raise ValueError('channels must contain at least input and output dimensions')
        self.gate_input_dim = gate_input_dim
        self.route_deep_layers = route_deep_layers
        self.layers = ModuleList([
            MoELayer(
                channels[i - 1],
                channels[i],
                num_experts=num_experts,
                top_k=top_k,
                use_norm=use_norm,
                gate_input_dim=gate_input_dim if i == 1 else None,
                router_noise=router_noise,
            )
            for i in range(1, len(channels))
        ])
        self.aux_loss = None

    def forward(self, x, gate_input=None):
        h = x
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:
                routing = gate_input
            elif self.route_deep_layers:
                routing = h
            else:
                routing = gate_input
            h = layer(h, gate_input=routing)
        return h


def collect_moe_load_balance_loss(module):
    """Sum auxiliary load-balancing losses from each MoE layer once."""
    total = None
    for submodule in module.modules():
        if isinstance(submodule, MoELayer) and submodule.aux_loss is not None:
            total = submodule.aux_loss if total is None else total + submodule.aux_loss
    if total is None:
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        return torch.tensor(0.0, device=device)
    return total


def collect_moe_expert_usage(module):
    """Average normalized expert usage across all MoE layers."""
    usages = []
    for submodule in module.modules():
        if isinstance(submodule, MoELayer) and submodule.expert_usage is not None:
            usages.append(submodule.expert_usage)
    if not usages:
        return None
    return torch.stack(usages, dim=0).mean(dim=0)


def collect_moe_router_entropy(module):
    """Average router entropy across MoE layers (lower => sharper routing)."""
    entropies = []
    for submodule in module.modules():
        if isinstance(submodule, MoELayer) and submodule.router_entropy is not None:
            entropies.append(submodule.router_entropy)
    if not entropies:
        return None
    return torch.stack(entropies).mean()


def collect_moe_train_metrics(module):
    """Return MoE diagnostics for logging."""
    metrics = {}
    lb_loss = collect_moe_load_balance_loss(module)
    if torch.is_tensor(lb_loss):
        metrics['load_balance_loss'] = float(lb_loss.detach().item())
    expert_usage = collect_moe_expert_usage(module)
    if expert_usage is not None:
        metrics['expert_usage_spread'] = float(
            (expert_usage.max() - expert_usage.min()).item()
        )
    router_entropy = collect_moe_router_entropy(module)
    if router_entropy is not None:
        metrics['router_entropy'] = float(router_entropy.item())
    return metrics
