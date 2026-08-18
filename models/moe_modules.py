#-------------------------------------------------------------------------------
# Name:        moe_modules.py
# Purpose:     reusable Mixture-of-Experts layers for RigNet models
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import torch
from torch.nn import Sequential, Linear, ReLU, LayerNorm, Module, ModuleList


# Shared expert keeps an always-active MLP path; routed experts add specialization.
MOE_NUM_EXPERTS = 4
MOE_TOP_K = 2
MOE_ROUTER_NOISE = 0.0
MOE_ROUTER_TEMPERATURE = 0.5
MOE_GATE_HIDDEN_RATIO = 4
MOE_USE_SHARED_EXPERT = True

MOE_NUM_EXPERTS_BONENET = 4
MOE_TOP_K_BONENET = 2


def _expert_block(in_channels, out_channels, use_norm=True):
    layers = [Linear(in_channels, out_channels), ReLU()]
    if use_norm:
        layers.append(LayerNorm(out_channels))
    return Sequential(*layers)


class MoEGate(Module):
    """MLP gate with normalized routing features and symmetricity breaking."""

    def __init__(self, in_channels, num_experts, hidden_ratio=MOE_GATE_HIDDEN_RATIO):
        super(MoEGate, self).__init__()
        hidden_dim = max(in_channels // hidden_ratio, num_experts * 2, 16)
        self.input_norm = LayerNorm(in_channels)
        self.net = Sequential(
            Linear(in_channels, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_experts),
        )
        self._reset_parameters(num_experts)

    def _reset_parameters(self, num_experts):
        for module in self.net:
            if isinstance(module, Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
                torch.nn.init.zeros_(module.bias)
        # Break expert symmetry so routing does not start perfectly uniform.
        last_linear = self.net[-1]
        torch.nn.init.normal_(last_linear.bias, mean=0.0, std=0.1)

    def forward(self, gate_features):
        gate_features = self.input_norm(gate_features)
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
    """Shared expert + top-k routed experts. Output = shared(x) + routed(x)."""

    def __init__(self, in_channels, out_channels, num_experts=MOE_NUM_EXPERTS, top_k=MOE_TOP_K,
                 use_norm=True, gate_input_dim=None, router_noise=MOE_ROUTER_NOISE,
                 router_temperature=MOE_ROUTER_TEMPERATURE, use_shared_expert=MOE_USE_SHARED_EXPERT):
        super(MoELayer, self).__init__()
        if num_experts < 1:
            raise ValueError('num_experts must be >= 1')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router_noise = router_noise
        self.router_temperature = router_temperature
        self.use_shared_expert = use_shared_expert
        self.gate_input_dim = gate_input_dim if gate_input_dim is not None else in_channels
        self.gate = MoEGate(self.gate_input_dim, num_experts)
        self.shared_expert = (
            _expert_block(in_channels, out_channels, use_norm=use_norm)
            if use_shared_expert else None
        )
        self.routed_experts = ModuleList([
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
        router_logits = router_logits / max(float(self.router_temperature), 1e-3)
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

        if self.shared_expert is not None:
            out = self.shared_expert(x)
        else:
            out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)

        for expert_idx, expert in enumerate(self.routed_experts):
            for k in range(self.top_k):
                mask = top_k_indices[:, k] == expert_idx
                if mask.any():
                    expert_out = expert(x[mask])
                    out[mask] = out[mask] + top_k_probs[mask, k:k + 1] * expert_out
        return out


class MoEMLP(Module):
    """Multi-layer MoE stack. Uses the same geometric gate features at every layer."""

    def __init__(self, channels, num_experts=MOE_NUM_EXPERTS, top_k=MOE_TOP_K, use_norm=True,
                 gate_input_dim=None, router_noise=MOE_ROUTER_NOISE,
                 router_temperature=MOE_ROUTER_TEMPERATURE, use_shared_expert=MOE_USE_SHARED_EXPERT):
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
                gate_input_dim=gate_input_dim,
                router_noise=router_noise,
                router_temperature=router_temperature,
                use_shared_expert=use_shared_expert,
            )
            for i in range(1, len(channels))
        ])

    def forward(self, x, gate_input=None):
        h = x
        for layer in self.layers:
            h = layer(h, gate_input=gate_input)
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
    """Average normalized routed-expert usage across all MoE layers."""
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
