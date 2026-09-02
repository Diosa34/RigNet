#-------------------------------------------------------------------------------
# MLflow metric helpers aligned with baseline RigNet naming conventions.
#-------------------------------------------------------------------------------


class BestTracker:
    """Track best validation metrics across epochs."""

    def __init__(self, lower_better=None, higher_better=None):
        self.lower_better = set(lower_better or [])
        self.higher_better = set(higher_better or [])
        self.values = {}
        self.epochs = {}

    def update(self, metrics, loss, epoch):
        if "loss" not in self.values or loss < self.values["loss"]:
            self.values["loss"] = loss
            self.epochs["loss"] = epoch
        for key, val in metrics.items():
            if key in self.lower_better:
                if key not in self.values or val < self.values[key]:
                    self.values[key] = val
                    self.epochs[key] = epoch
            elif key in self.higher_better:
                if key not in self.values or val > self.values[key]:
                    self.values[key] = val
                    self.epochs[key] = epoch

    def log_summary(self, prefix="best_val"):
        import mlflow
        for key, val in self.values.items():
            mlflow.log_metric(f"{prefix}_{key}", float(val))
            if key in self.epochs:
                mlflow.log_param(f"{prefix}_{key}_epoch", int(self.epochs[key]))


def log_mask_f1_steps(prefix, step_metrics, epoch):
    """Diagnostic per-step F1: mask_f1_step0 .. mask_f1_stepT."""
    import mlflow
    if not step_metrics or "per_step" not in step_metrics:
        return
    for t, sm in step_metrics["per_step"].items():
        if "mask_f1" in sm:
            mlflow.log_metric(f"{prefix}_mask_f1_step{t}", sm["mask_f1"], step=epoch)


def log_gate_entropy_steps(prefix, step_metrics, epoch, num_experts):
    """MoR diagnostic: gate_entropy_step{t}."""
    import mlflow
    if num_experts <= 1 or not step_metrics:
        return
    for t, tr in step_metrics.get("per_transition", {}).items():
        if "gate_entropy" in tr:
            mlflow.log_metric(f"{prefix}_gate_entropy_step{t}", tr["gate_entropy"], step=epoch)


def compute_gate_entropy_mean(step_metrics):
    transitions = step_metrics.get("per_transition", {}) if step_metrics else {}
    entropies = [tr["gate_entropy"] for tr in transitions.values() if "gate_entropy" in tr]
    return float(sum(entropies) / len(entropies)) if entropies else 0.0
