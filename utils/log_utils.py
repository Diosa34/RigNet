#-------------------------------------------------------------------------------
# Name:        mst_utils.py
# Purpose:     utilize class for log recording
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def load_state_dict_compat(model, state_dict, prefix=''):
    """Load checkpoint with strict=False and log missing/unexpected keys."""
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    label = f'{prefix}: ' if prefix else ''
    if missing:
        print(f'=> {label}missing keys ({len(missing)}): {missing}')
    if unexpected:
        print(f'=> {label}unexpected keys ({len(unexpected)}): {unexpected}')
    if not missing and not unexpected:
        print(f'=> {label}all keys matched')
    return missing, unexpected


def count_mha_block_params(model):
    """Count MHA block parameters split by submodule."""
    mha_params = 0
    layernorm_params = 0
    mha = getattr(model, 'mha', None)
    mha_norm = getattr(model, 'mha_norm', None)
    if mha is not None:
        mha_params = sum(p.numel() for p in mha.parameters())
    if mha_norm is not None:
        layernorm_params = sum(p.numel() for p in mha_norm.parameters())
    return {
        'mha_params': mha_params,
        'layernorm_params': layernorm_params,
        'mha_block_params': mha_params + layernorm_params,
    }


def log_best_test_metrics(metrics, inference_time_sec=None):
    """Log final best-checkpoint test metrics to MLflow with best_test_ prefix."""
    joint_metric_map = {
        'mean_joint_error': 'best_test_mean_joint_error',
        'median_joint_error': 'best_test_median_joint_error',
        'joint_precision': 'best_test_joint_precision',
        'joint_recall': 'best_test_joint_recall',
        'joint_f1': 'best_test_joint_f1',
        'pred_joint_count': 'best_test_pred_joint_count',
        'gt_joint_count': 'best_test_gt_joint_count',
        'joint_count_error': 'best_test_joint_count_error',
        'mask_precision': 'best_test_mask_precision',
        'mask_recall': 'best_test_mask_recall',
        'mask_f1': 'best_test_mask_f1',
        'precision': 'best_test_precision',
        'recall': 'best_test_recall',
        'f1': 'best_test_f1',
        'pr_auc': 'best_test_pr_auc',
        'loss': 'best_test_loss',
    }
    for src_key, dst_key in joint_metric_map.items():
        if src_key in metrics:
            mlflow.log_metric(dst_key, float(metrics[src_key]))
    if inference_time_sec is not None:
        mlflow.log_metric('best_test_inference_time_sec', float(inference_time_sec))


def setup_device(gpu_id=0):
    """Select torch device; use cuda:{gpu_id} when CUDA is available."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    return device