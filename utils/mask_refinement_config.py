#-------------------------------------------------------------------------------
# Unified argparse / preset configuration for iterative mask refinement ablations.
# User-facing refinement settings: num_refine_steps, num_experts, lambda_intermediate.
# shared_refinement is fixed True for all main ablations.
#-------------------------------------------------------------------------------
import argparse

SHARED_REFINEMENT = True

ABLATION_PRESETS = {
    "iterative-2-shared": {"num_refine_steps": 2, "num_experts": 1},
    "iterative-4-shared": {"num_refine_steps": 4, "num_experts": 1},
    "iterative-2-2experts": {"num_refine_steps": 2, "num_experts": 2},
    "iterative-4-2experts": {"num_refine_steps": 4, "num_experts": 2},
}


def add_refinement_args(parser):
    parser.add_argument(
        "--ablation",
        default=None,
        choices=list(ABLATION_PRESETS.keys()),
        help="Named ablation preset (overrides num_refine_steps / num_experts)",
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
        help="Number of MoR refinement experts (1 = single expert, no gating)",
    )
    parser.add_argument(
        "--lambda-intermediate",
        dest="lambda_intermediate",
        default=0.25,
        type=float,
        help="Weight for mean intermediate mask BCE (steps 1..T-1, excludes step0 and final)",
    )


def apply_ablation_preset(args):
    if args.ablation is None:
        args.shared_refinement = SHARED_REFINEMENT
        return args
    preset = ABLATION_PRESETS[args.ablation]
    args.num_refine_steps = preset["num_refine_steps"]
    args.num_experts = preset["num_experts"]
    args.shared_refinement = SHARED_REFINEMENT
    return args
