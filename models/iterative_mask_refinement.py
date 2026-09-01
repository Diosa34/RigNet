#-------------------------------------------------------------------------------
# Iterative mask / attention refinement with optional Mixture-of-Recursion (MoR).
# Updates logits via residual deltas: logits_{t+1} = logits_t + Δlogits_t.
#-------------------------------------------------------------------------------
import torch
from torch.nn import Module, ModuleList, Sequential, Linear, ReLU


def _assert_batch_offsets(batch, num_vertices):
    """Validate data.batch offsets for variable-size batched meshes."""
    assert batch.shape[0] == num_vertices, (
        f"batch length {batch.shape[0]} != num_vertices {num_vertices}"
    )
    assert batch.dim() == 1, f"batch must be 1-D, got shape {batch.shape}"
    assert batch.dtype in (torch.int32, torch.int64), f"batch dtype {batch.dtype}"
    assert int(batch.min()) >= 0, "batch indices must be non-negative"
    counts = torch.bincount(batch)
    assert int(counts.sum()) == num_vertices, "batch bincount sum must equal num_vertices"
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    for graph_id in range(counts.numel()):
        mask = batch == graph_id
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        assert int(idx.min()) == int(offsets[graph_id]), (
            f"graph {graph_id}: vertex indices not contiguous in batch"
        )


def _assert_finite(name, tensor):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


def _assert_vertex_consistent(*tensors):
    n = tensors[0].shape[0]
    for t in tensors[1:]:
        assert t.shape[0] == n, "all per-vertex tensors must share the same vertex count"


class RefinementExpert(Module):
    """Lightweight per-vertex expert: [h, displacement, attention] -> Δlogits."""

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

    def forward(self, h, displacement, attention):
        x = torch.cat([h, displacement, attention], dim=1)
        return self.mlp(x)


class GatingNetwork(Module):
    """Per-vertex softmax gating over refinement experts."""

    def __init__(self, h_dim, num_experts, hidden_dim):
        super().__init__()
        in_dim = h_dim + 3 + 1
        self.mlp = Sequential(
            Linear(in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_experts),
        )
        self.num_experts = num_experts

    def forward(self, h, displacement, attention, batch):
        _assert_batch_offsets(batch, h.shape[0])
        x = torch.cat([h, displacement, attention], dim=1)
        gate_logits = self.mlp(x)
        assert gate_logits.shape == (h.shape[0], self.num_experts)
        return torch.softmax(gate_logits, dim=1)


class IterativeMaskRefinement(Module):
    """
    Iteratively refines mask logits after the initial MaskNet prediction.

    JointNet displacement is fixed across steps (q_t = pos + displacement).
    Persistent vertex features h are reused at every step.
    """

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
        assert num_refine_steps >= 1, "num_refine_steps must be >= 1"
        assert num_experts >= 1, "num_experts must be >= 1"

        self.h_dim = h_dim
        self.num_refine_steps = num_refine_steps
        self.num_experts = num_experts
        self.shared_refinement = shared_refinement

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
                for _ in range(num_refine_steps)
            ])
            self.step_gatings = ModuleList([
                GatingNetwork(h_dim, num_experts, gating_hidden_dim) if num_experts > 1 else None
                for _ in range(num_refine_steps)
            ])

    def _modules_for_step(self, step_idx):
        if self.shared_refinement:
            return self.experts, self.gating
        return self.step_experts[step_idx], self.step_gatings[step_idx]

    def _compute_delta_logits(self, experts, gating, h, displacement, attention, batch):
        delta_per_expert = [expert(h, displacement, attention) for expert in experts]
        for delta in delta_per_expert:
            assert delta.shape == (h.shape[0], 1)

        if self.num_experts == 1:
            return delta_per_expert[0], None

        gate_weights = gating(h, displacement, attention, batch)
        delta_stack = torch.stack(delta_per_expert, dim=1)
        assert delta_stack.shape == (h.shape[0], self.num_experts, 1)
        delta_logits = (gate_weights.unsqueeze(-1) * delta_stack).sum(dim=1)
        return delta_logits, gate_weights

    def forward(self, h, displacement, logits_0, batch, num_refine_steps=None):
        steps = num_refine_steps if num_refine_steps is not None else self.num_refine_steps
        assert steps >= 1

        _assert_vertex_consistent(h, displacement, logits_0)
        assert h.shape[1] == self.h_dim, f"h dim {h.shape[1]} != expected {self.h_dim}"
        assert displacement.shape[1] == 3
        assert logits_0.shape[1] == 1
        _assert_batch_offsets(batch, h.shape[0])
        _assert_finite("h", h)
        _assert_finite("displacement", displacement)
        _assert_finite("logits_0", logits_0)

        logits_steps = [logits_0]
        attention_steps = [torch.sigmoid(logits_0)]
        delta_logits_steps = []
        gate_entropy_steps = []

        for t in range(steps):
            logits_t = logits_steps[-1]
            attention_t = attention_steps[-1]
            experts, gating = self._modules_for_step(t)

            delta_logits_t, gate_weights = self._compute_delta_logits(
                experts, gating, h, displacement, attention_t, batch
            )
            _assert_finite(f"delta_logits_step_{t}", delta_logits_t)

            logits_next = logits_t + delta_logits_t
            _assert_finite(f"logits_step_{t + 1}", logits_next)
            _assert_vertex_consistent(logits_next, logits_t, attention_t)

            logits_steps.append(logits_next)
            attention_steps.append(torch.sigmoid(logits_next))
            delta_logits_steps.append(delta_logits_t)

            if gate_weights is not None:
                entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(dim=1).mean()
                gate_entropy_steps.append(entropy)

        return {
            "logits_steps": logits_steps,
            "attention_steps": attention_steps,
            "delta_logits_steps": delta_logits_steps,
            "gate_entropy_steps": gate_entropy_steps,
        }
