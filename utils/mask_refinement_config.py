#-------------------------------------------------------------------------------
# Unified argparse / preset configuration for iterative mask refinement ablations.
# User-facing refinement settings: num_refine_steps, num_experts, lambda_intermediate.
# shared_refinement is fixed True for all main ablations.
#-------------------------------------------------------------------------------

SHARED_REFINEMENT = True

ABLATION_PRESETS = {
    # A0: no refinement at all -> reproduces the pretrained MaskNet prediction exactly.
    "reference_0": {"num_refine_steps": 0, "num_experts": 1},
    "iterative_1_shared": {"num_refine_steps": 1, "num_experts": 1},
    "iterative_2_shared": {"num_refine_steps": 2, "num_experts": 1},
    "iterative_4_shared": {"num_refine_steps": 4, "num_experts": 1},
    "iterative_2_2experts": {"num_refine_steps": 2, "num_experts": 2},
    "iterative_4_2experts": {"num_refine_steps": 4, "num_experts": 2},
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
        help="Number of iterative mask refinement steps after initial MaskNet prediction (0 = none)",
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
        default=0.05,
        type=float,
        help="Weight for mean intermediate mask BCE (steps 1..T-1); must not exceed --bce_loss_weight",
    )
    parser.add_argument(
        "--no-freeze-masknet",
        dest="freeze_masknet",
        action="store_false",
        default=True,
        help="Finetune MaskNet alongside refinement instead of keeping it frozen",
    )
    parser.add_argument(
        "--refine-debug",
        dest="refine_debug",
        action="store_true",
        help="Enable per-forward shape/finiteness assertions in the refinement module",
    )
    return parser


def apply_ablation_preset(args):
    if args.ablation is not None:
        preset = ABLATION_PRESETS[args.ablation]
        args.num_refine_steps = preset["num_refine_steps"]
        args.num_experts = preset["num_experts"]
    args.shared_refinement = SHARED_REFINEMENT

    if args.num_refine_steps < 0:
        raise ValueError("num_refine_steps must be >= 0")

    if getattr(args, "use_bce", False):
        bce_weight = getattr(args, "bce_loss_weight", 0.0)
        if args.lambda_intermediate > bce_weight:
            raise ValueError(
                "lambda_intermediate ({:g}) must not exceed bce_loss_weight ({:g}); "
                "otherwise intermediate steps are supervised more strongly than the "
                "final mask that mean-shift actually consumes".format(
                    args.lambda_intermediate, bce_weight
                )
            )
    return args
