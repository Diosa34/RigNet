#-------------------------------------------------------------------------------
# Unified argparse / preset configuration for iterative mask refinement ablations.
#-------------------------------------------------------------------------------
import argparse


ABLATION_PRESETS = {
    "iterative-2-shared": {
        "num_refine_steps": 2,
        "num_experts": 1,
        "shared_refinement": True,
    },
    "iterative-4-shared": {
        "num_refine_steps": 4,
        "num_experts": 1,
        "shared_refinement": True,
    },
    "iterative-2-2experts": {
        "num_refine_steps": 2,
        "num_experts": 2,
        "shared_refinement": True,
    },
    "iterative-4-2experts": {
        "num_refine_steps": 4,
        "num_experts": 2,
        "shared_refinement": True,
    },
}


def add_refinement_args(parser):
    """Register core refinement hyper-parameters on an argparse parser."""
    parser.add_argument(
        "--ablation",
        default=None,
        choices=list(ABLATION_PRESETS.keys()),
        help="Named ablation preset overriding num_refine_steps / num_experts / shared_refinement",
    )
    parser.add_argument(
        "--num-refine-steps",
        dest="num_refine_steps",
        default=2,
        type=int,
        help="Number of iterative mask refinement steps after initial MaskNet prediction",
    )
    parser.add_argument(
        "--num-experts",
        dest="num_experts",
        default=1,
        type=int,
        choices=[1, 2, 3],
        help="Number of MoR refinement experts (1 = single shared expert, no gating)",
    )
    parser.add_argument(
        "--shared-refinement",
        dest="shared_refinement",
        action="store_true",
        default=True,
        help="Share refinement expert / gating weights across steps (default: True)",
    )
    parser.add_argument(
        "--no-shared-refinement",
        dest="shared_refinement",
        action="store_false",
        help="Use separate refinement weights per step",
    )
    parser.add_argument(
        "--lambda-intermediate",
        dest="lambda_intermediate",
        default=0.25,
        type=float,
        help="Weight for mean intermediate mask BCE across refinement steps",
    )
    parser.add_argument(
        "--refine-hidden-dim",
        dest="refine_hidden_dim",
        default=256,
        type=int,
        help="Hidden width of each refinement expert MLP",
    )
    parser.add_argument(
        "--gating-hidden-dim",
        dest="gating_hidden_dim",
        default=128,
        type=int,
        help="Hidden width of the MoR gating network",
    )
    parser.add_argument(
        "--refine-lr",
        dest="refine_lr",
        default=5e-5,
        type=float,
        help="Learning rate for IterativeMaskRefinement parameters",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility",
    )


def log_experiment_config(args):
    """Log core experiment configuration as MLflow params."""
    import mlflow
    for key in (
        "ablation", "num_refine_steps", "num_experts", "shared_refinement",
        "lambda_intermediate", "refine_hidden_dim", "gating_hidden_dim",
        "refine_lr", "masknet_lr", "bandwidth_lr", "bce_loss_weight",
        "ms_loss_weight", "seed", "epochs", "use_bce",
    ):
        if hasattr(args, key):
            mlflow.log_param(key, getattr(args, key))
    mlflow.log_param("jointnet_frozen", True)


def apply_ablation_preset(args):
    """Apply a named ablation preset to args in-place."""
    if args.ablation is None:
        return args
    preset = ABLATION_PRESETS[args.ablation]
    args.num_refine_steps = preset["num_refine_steps"]
    args.num_experts = preset["num_experts"]
    args.shared_refinement = preset["shared_refinement"]
    return args
