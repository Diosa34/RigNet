#-------------------------------------------------------------------------------
# Iterative mask / attention refinement with optional Mixture-of-Recursion (MoR).
# logits_{t+1} = logits_t + alpha_t * tanh(raw_delta_t)
# attention_t = sigmoid(logits_t) — never updated directly.
# Each step: q_t = pos + displacement_t; experts receive [LayerNorm(h), q_t - pos, attention_t].
# The delta head is zero-initialised, so refinement is the identity at initialisation
# and num_refine_steps=0 reproduces the plain MaskNet prediction exactly.
#-------------------------------------------------------------------------------
import os
import torch
from torch.nn import Module, ModuleList, Sequential, Linear, ReLU, LayerNorm, Parameter

# Shape/finiteness invariants are expensive on every forward (nonzero scans per graph),
# so they are opt-in via --refine_debug or REFINE_DEBUG=1.
_DEBUG = os.environ.get("REFINE_DEBUG", "0") == "1"


def set_debug(enabled):
    global _DEBUG
    _DEBUG = bool(enabled)


def debug_enabled():
    return _DEBUG


def _assert_batch_offsets(batch, num_vertices):
    if not _DEBUG:
        return
    assert batch.shape[0] == num_vertices
    assert batch.dim() == 1
    assert batch.dtype in (torch.int32, torch.int64)
    assert int(batch.min()) >= 0
    counts = torch.bincount(batch)
    assert int(counts.sum()) == num_vertices
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    for graph_id in range(counts.numel()):
        mask = batch == graph_id
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        assert int(idx.min()) == int(offsets[graph_id])


def _assert_finite(name, tensor):
    if not _DEBUG:
        return
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


def _assert_vertex_consistent(*tensors):
    if not _DEBUG:
        return
    n = tensors[0].shape[0]
    for t in tensors[1:]:
        assert t.shape[0] == n, "all per-vertex tensors must share the same vertex count"


def _assert_q_t(q_t, pos, displacement_t):
    if not _DEBUG:
        return
    _assert_vertex_consistent(q_t, pos, displacement_t)
    assert q_t.shape[1] == 3 and pos.shape[1] == 3 and displacement_t.shape[1] == 3
    _assert_finite("q_t", q_t)
    assert torch.allclose(q_t, pos + displacement_t, atol=1e-5, rtol=1e-5)


def _assert_attention_from_logits(logits, attention, step_label=""):
    if not _DEBUG:
        return
    assert logits.shape == attention.shape
    expected = torch.sigmoid(logits)
    assert torch.allclose(attention, expected, atol=1e-5, rtol=1e-5), (
        f"attention must equal sigmoid(logits) at {step_label}"
    )
    assert attention.min() >= 0.0 and attention.max() <= 1.0


def _assert_gate_weights(gate_weights, num_experts):
    """Per-vertex softmax over expert dimension; no cross-graph mixing in gating."""
    if not _DEBUG:
        return
    assert gate_weights.dim() == 2
    assert gate_weights.shape[1] == num_experts
    _assert_finite("gate_weights", gate_weights)
    sums = gate_weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
        "gate softmax must sum to 1 per vertex"
    )
    assert gate_weights.min() >= 0.0 and gate_weights.max() <= 1.0


class RefinementExpert(Module):
    """Per-vertex expert: [h, q_t - pos, attention_t] -> raw delta logit."""

    def __init__(self, h_dim, hidden_dim):
        super().__init__()
        in_dim = h_dim + 3 + 1
        self.mlp = Sequential(
            Linear(in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1),
        )
        # Identity at initialisation: the refined logits start exactly at MaskNet's.
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, rel_displacement, attention):
        assert rel_displacement.shape[1] == 3 and attention.shape[1] == 1
        x = torch.cat([h, rel_displacement, attention], dim=1)
        return self.mlp(x)


class GatingNetwork(Module):
    """Per-vertex softmax gating; each vertex gates independently (no graph mixing)."""

    def __init__(self, h_dim, num_experts, hidden_dim):
        super().__init__()
        in_dim = h_dim + 3 + 1
        self.mlp = Sequential(
            Linear(in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_experts),
        )
        self.num_experts = num_experts

    def forward(self, h, rel_displacement, attention, batch):
        _assert_batch_offsets(batch, h.shape[0])
        x = torch.cat([h, rel_displacement, attention], dim=1)
        gate_logits = self.mlp(x)
        assert gate_logits.shape == (h.shape[0], self.num_experts)
        gate_weights = torch.softmax(gate_logits, dim=1)
        _assert_gate_weights(gate_weights, self.num_experts)
        return gate_weights


class IterativeMaskRefinement(Module):
    """
    MaskNet logits refinement with optional Mixture-of-Recursion.

    State at step t:
      q_t = pos + displacement_t   (displacement_t = displacement_0 when JointNet frozen)
      rel = q_t - pos
      input = [LayerNorm(h), rel, attention_t]
      logits_{t+1} = logits_t + alpha_t * tanh(raw_delta_t)
      attention_{t+1} = sigmoid(logits_{t+1})

    LayerNorm decouples the experts from MaskNet's BatchNorm train/eval statistics;
    alpha_t damps the residual so that stacking steps cannot blow the logits up.
    """

    VALID_NUM_EXPERTS = (1, 2, 3)
    ALPHA_INIT = 0.1

    def __init__(
        self,
        h_dim=832,
        num_refine_steps=2,
        num_experts=1,
        shared_refinement=True,
        refine_hidden_dim=256,
        gating_hidden_dim=128,
    ):
        super().__init__()
        assert num_refine_steps >= 0
        assert num_experts in self.VALID_NUM_EXPERTS, (
            f"num_experts must be one of {self.VALID_NUM_EXPERTS}, got {num_experts}"
        )

        self.h_dim = h_dim
        self.num_refine_steps = num_refine_steps
        self.num_experts = num_experts
        self.shared_refinement = shared_refinement

        self.h_norm = LayerNorm(h_dim)
        self.alpha = Parameter(torch.full((max(num_refine_steps, 1),), self.ALPHA_INIT))

        if shared_refinement:
            self.experts = ModuleList([
                RefinementExpert(h_dim, refine_hidden_dim) for _ in range(num_experts)
            ])
            self.gating = GatingNetwork(h_dim, num_experts, gating_hidden_dim) if num_experts > 1 else None
            self.step_experts = None
            self.step_gatings = None
        else:
            self.experts = None
            self.gating = None
            self.step_experts = ModuleList([
                ModuleList([RefinementExpert(h_dim, refine_hidden_dim) for _ in range(num_experts)])
                for _ in range(max(num_refine_steps, 1))
            ])
            self.step_gatings = ModuleList([
                GatingNetwork(h_dim, num_experts, gating_hidden_dim) if num_experts > 1 else None
                for _ in range(max(num_refine_steps, 1))
            ])

    def _modules_for_step(self, step_idx):
        if self.shared_refinement:
            return self.experts, self.gating
        idx = min(step_idx, len(self.step_experts) - 1)
        return self.step_experts[idx], self.step_gatings[idx]

    def _compute_raw_delta(self, experts, gating, h, rel_displacement, attention, batch):
        delta_per_expert = [expert(h, rel_displacement, attention) for expert in experts]
        if self.num_experts == 1:
            return delta_per_expert[0], None

        gate_weights = gating(h, rel_displacement, attention, batch)
        delta_stack = torch.stack(delta_per_expert, dim=1)
        assert delta_stack.shape == (h.shape[0], self.num_experts, 1)
        return (gate_weights.unsqueeze(-1) * delta_stack).sum(dim=1), gate_weights

    def forward(self, h, pos, displacement_0, logits_0, batch, num_refine_steps=None):
        steps = num_refine_steps if num_refine_steps is not None else self.num_refine_steps
        assert steps >= 0

        _assert_vertex_consistent(h, pos, displacement_0, logits_0)
        assert h.shape[1] == self.h_dim
        assert logits_0.shape[1] == 1
        _assert_batch_offsets(batch, h.shape[0])
        for name, t in [("h", h), ("pos", pos), ("displacement_0", displacement_0), ("logits_0", logits_0)]:
            _assert_finite(name, t)

        h_normed = self.h_norm(h)

        q_0 = pos + displacement_0
        _assert_q_t(q_0, pos, displacement_0)

        logits_steps = [logits_0]
        attention_steps = [torch.sigmoid(logits_0)]
        _assert_attention_from_logits(logits_0, attention_steps[0], "step0")

        displacement_steps = [displacement_0]
        q_steps = [q_0]
        delta_logits_steps = []
        gate_entropy_steps = []
        prob_shift_steps = []
        alpha_steps = []

        for t in range(steps):
            logits_t = logits_steps[-1]
            attention_t = attention_steps[-1]
            displacement_t = displacement_steps[-1]

            q_t = pos + displacement_t
            _assert_q_t(q_t, pos, displacement_t)
            rel_displacement = q_t - pos
            _assert_finite(f"rel_displacement_step_{t}", rel_displacement)

            experts, gating = self._modules_for_step(t)
            raw_delta_t, gate_weights = self._compute_raw_delta(
                experts, gating, h_normed, rel_displacement, attention_t, batch
            )
            alpha_t = self.alpha[min(t, self.alpha.numel() - 1)]
            delta_logits_t = alpha_t * torch.tanh(raw_delta_t)
            _assert_finite(f"delta_logits_step_{t}", delta_logits_t)
            assert delta_logits_t.shape == (h.shape[0], 1)

            logits_next = logits_t + delta_logits_t
            attention_next = torch.sigmoid(logits_next)
            _assert_finite(f"logits_step_{t + 1}", logits_next)
            _assert_attention_from_logits(logits_next, attention_next, f"step{t + 1}")
            _assert_vertex_consistent(logits_next, logits_t, attention_t, q_t, h)

            prob_shift_steps.append((attention_next - attention_t).abs().mean())
            logits_steps.append(logits_next)
            attention_steps.append(attention_next)
            delta_logits_steps.append(delta_logits_t)
            alpha_steps.append(alpha_t.detach())

            displacement_next = displacement_0
            q_next = pos + displacement_next
            _assert_q_t(q_next, pos, displacement_next)
            displacement_steps.append(displacement_next)
            q_steps.append(q_next)

            if gate_weights is not None:
                entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(dim=1).mean()
                _assert_finite(f"gate_entropy_step_{t}", entropy.unsqueeze(0))
                gate_entropy_steps.append(entropy)

        assert len(logits_steps) == steps + 1
        assert len(attention_steps) == steps + 1
        assert len(q_steps) == steps + 1

        return {
            "logits_steps": logits_steps,
            "attention_steps": attention_steps,
            "displacement_steps": displacement_steps,
            "q_steps": q_steps,
            "delta_logits_steps": delta_logits_steps,
            "gate_entropy_steps": gate_entropy_steps,
            "prob_shift_steps": prob_shift_steps,
            "alpha_steps": alpha_steps,
        }
