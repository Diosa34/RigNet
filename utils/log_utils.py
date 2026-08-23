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


def count_mha_params(model):
    """Count parameters in MHA + LayerNorm blocks (if present)."""
    total = 0
    for name in ('mha', 'mha_norm'):
        module = getattr(model, name, None)
        if module is not None:
            total += sum(p.numel() for p in module.parameters())
    return total


def setup_device(gpu_id=0):
    """Select torch device; use cuda:{gpu_id} when CUDA is available."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    return device