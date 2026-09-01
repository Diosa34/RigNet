#-------------------------------------------------------------------------------
# Iterative mask / attention refinement with optional Mixture-of-Recursion (MoR).
# Updates logits via residual deltas: logits_{t+1} = logits_t + Δlogits_t.
# Each step explicitly forms q_t = pos + displacement_t and passes q_t - pos.
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


def _assert_q_t(q_t, pos, displacement_t):
    """Validate geometric hypothesis q_t = pos + displacement_t."""
    _assert_vertex_consistent(q_t, pos, displacement_t)
    assert q_t.shape[1] == 3 and pos.shape[1] == 3 and displacement_t.shape[1] == 3
    _assert_finite("q_t", q_t)
    expected = pos + displacement_t
    assert torch.allclose(q_t, expected, atol=1e-5, rtol=1e-5), (
        "q_t must equal pos + displacement_t"
    )


class RefinementExpert(Module):
    """Lightweight per-vertex expert: [h, q_t - pos, attention] -> Δlogits."""

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

    def forward(self, h, rel_displacement, attention):
        x = torch.cat([h, rel_displacement, attention], dim=1)
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

    def forward(self, h, rel_displacement, attention, batch):
        _assert_batch_offsets(batch, h.shape[0])
        x = torch.cat([h, rel_displacement, attention], dim=1)
        gate_logits = self.mlp(x)
        assert gate_logits.shape == (h.shape[0], self.num_experts)
        return torch.softmax(gate_logits, dim=1)


class IterativeMaskRefinement(Module):
    """
    Iteratively refines mask logits after the initial MaskNet prediction.

    At each step t the geometric hypothesis is q_t = pos + displacement_t.
    Experts receive [h, q_t - pos, attention_t].  For attention-only refinement
    displacement_t stays equal to the initial JointNet displacement.
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

    def _compute_delta_logits(self, experts, gating, h, rel_displacement, attention, batch):
        delta_per_expert = [expert(h, rel_displacement, attention) for expert in experts]
        for delta in delta_per_expert:
            assert delta.shape == (h.shape[0], 1)

        if self.num_experts == 1:
            return delta_per_expert[0], None

        gate_weights = gating(h, rel_displacement, attention, batch)
        delta_stack = torch.stack(delta_per_expert, dim=1)
        assert delta_stack.shape == (h.shape[0], self.num_experts, 1)
        delta_logits = (gate_weights.unsqueeze(-1) * delta_stack).sum(dim=1)
        return delta_logits, gate_weights

    def forward(self, h, pos, displacement_0, logits_0, batch, num_refine_steps=None):
        steps = num_refine_steps if num_refine_steps is not None else self.num_refine_steps
        assert steps >= 1

        _assert_vertex_consistent(h, pos, displacement_0, logits_0)
        assert h.shape[1] == self.h_dim, f"h dim {h.shape[1]} != expected {self.h_dim}"
        assert pos.shape[1] == 3 and displacement_0.shape[1] == 3
        assert logits_0.shape[1] == 1
        _assert_batch_offsets(batch, h.shape[0])
        _assert_finite("h", h)
        _assert_finite("pos", pos)
        _assert_finite("displacement_0", displacement_0)
        _assert_finite("logits_0", logits_0)

        q_0 = pos + displacement_0
        _assert_q_t(q_0, pos, displacement_0)

        logits_steps = [logits_0]
        attention_steps = [torch.sigmoid(logits_0)]
        displacement_steps = [displacement_0]
        q_steps = [q_0]
        delta_logits_steps = []
        gate_entropy_steps = []
        prob_shift_steps = []

        for t in range(steps):
            logits_t = logits_steps[-1]
            attention_t = attention_steps[-1]
            displacement_t = displacement_steps[-1]
            q_t = pos + displacement_t
            _assert_q_t(q_t, pos, displacement_t)

            rel_displacement = q_t - pos
            assert rel_displacement.shape == displacement_t.shape
            _assert_finite(f"rel_displacement_step_{t}", rel_displacement)

            experts, gating = self._modules_for_step(t)
            delta_logits_t, gate_weights = self._compute_delta_logits(
                experts, gating, h, rel_displacement, attention_t, batch
            )
            _assert_finite(f"delta_logits_step_{t}", delta_logits_t)

            logits_next = logits_t + delta_logits_t
            attention_next = torch.sigmoid(logits_next)
            _assert_finite(f"logits_step_{t + 1}", logits_next)
            _assert_vertex_consistent(logits_next, logits_t, attention_t, q_t)

            prob_shift_steps.append((attention_next - attention_t).abs().mean())

            logits_steps.append(logits_next)
            attention_steps.append(attention_next)
            delta_logits_steps.append(delta_logits_t)

            # Attention-only refinement: displacement stays fixed; q_t recomputed explicitly.
            displacement_next = displacement_0
            q_next = pos + displacement_next
            _assert_q_t(q_next, pos, displacement_next)
            displacement_steps.append(displacement_next)
            q_steps.append(q_next)

            if gate_weights is not None:
                entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(dim=1).mean()
                gate_entropy_steps.append(entropy)

        return {
            "logits_steps": logits_steps,
            "attention_steps": attention_steps,
            "displacement_steps": displacement_steps,
            "q_steps": q_steps,
            "delta_logits_steps": delta_logits_steps,
            "gate_entropy_steps": gate_entropy_steps,
            "prob_shift_steps": prob_shift_steps,
        }
