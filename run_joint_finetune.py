#-------------------------------------------------------------------------------
# Name:        run_joint_finetune.py
# Purpose:     Iterative MaskNet / MoR finetune (JointNet strictly frozen) + mean-shift
#-------------------------------------------------------------------------------

import sys
sys.path.append("./")
import os
import random
import numpy as np
import shutil
import argparse

from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import output_point_cloud_ply
from utils.log_args_to_mlflow import log_args_to_mlflow
from utils.mask_refinement_config import add_refinement_args, apply_ablation_preset, SHARED_REFINEMENT
from utils.mlflow_metrics import (
    BestTracker,
    log_mask_step_metrics,
    log_gate_entropy_steps,
    log_alpha_steps,
    compute_gate_entropy_mean,
)

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.GCN import JOINTNET_MASKNET_MEANSHIFT
from models.iterative_mask_refinement import set_debug as set_refine_debug
from datasets.skeleton_dataset import GraphDataset
from models.supplemental_layers.pytorch_chamfer_dist import chamfer_distance_with_average

import mlflow

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

REFINE_LR_DEFAULT = 5e-5  # refinement is trained from scratch, unlike the pretrained backbones
MEANSHIFT_METRICS = ("cd_after", "avg_shift_last", "total_disp", "disp_std")
REFINEMENT_METRICS = ("mask_f1_improvement", "refine_delta_logit_mean", "mask_prob_shift_mean")
# Logged separately so that runs with and without --use_bce remain comparable.
LOSS_COMPONENTS = ("base_loss", "bce_final", "bce_intermediate")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def verify_optimizer_scope(model, optimizer):
    optim_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    for p in model.jointnet.parameters():
        assert id(p) not in optim_ids, "JointNet must not be in the optimizer"
    if model._masknet_frozen:
        for p in model.masknet.parameters():
            assert id(p) not in optim_ids, "frozen MaskNet must not be in the optimizer"
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(id(p) in optim_ids for p in trainable), "some trainable params are not optimized"


def pairwise_distances(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def meanshift_cluster(pts, bandwidth, weights, args):
    pts_steps = []
    for _ in range(args.meanshift_step):
        Y = pairwise_distances(pts, pts)
        K = F.relu(bandwidth ** 2 - Y)
        if weights is not None:
            K = K * weights
        P = F.normalize(K, p=1, dim=0, eps=1e-10)
        P = P.transpose(0, 1)
        pts = args.step_size * (torch.matmul(P, pts) - pts) + pts
        pts_steps.append(pts)
    return pts_steps


def compute_mask_bce_split(logits_steps, mask_gt):
    """
    L_final = BCE(logits_T, mask_gt)
    L_intermediate = mean(BCE(logits_t, mask_gt) for t in 1..T-1)
    Step0 (initial MaskNet) and final step T are excluded from L_intermediate.
    For T <= 1: L_intermediate = 0.
    """
    mask_gt = mask_gt.float()
    T = len(logits_steps) - 1
    final_loss = F.binary_cross_entropy_with_logits(logits_steps[T], mask_gt, reduction='mean')

    if T <= 1:
        intermediate_loss = logits_steps[0].new_zeros(())
    else:
        intermediate_losses = [
            F.binary_cross_entropy_with_logits(logits_steps[t], mask_gt, reduction='mean')
            for t in range(1, T)
        ]
        intermediate_loss = sum(intermediate_losses) / len(intermediate_losses)

    return final_loss, intermediate_loss


def compute_step_classification(probs, labels):
    pred_bin = (probs > 0.5).astype(np.uint8)
    labels = labels.astype(np.uint8)
    return {
        "mask_precision": precision_score(labels, pred_bin, zero_division=0),
        "mask_recall": recall_score(labels, pred_bin, zero_division=0),
        "mask_f1": f1_score(labels, pred_bin, zero_division=0),
        "mask_pr_auc": average_precision_score(labels, probs),
    }


def collect_step_metrics(logits_steps, delta_logits_steps, prob_shift_steps,
                         gate_entropy_steps, mask_gt, num_experts):
    per_step = {}
    labels = mask_gt.detach().cpu().numpy().reshape(-1).astype(np.uint8)

    for t, logits in enumerate(logits_steps):
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        loss_t = F.binary_cross_entropy_with_logits(logits, mask_gt.float(), reduction='mean').item()
        cls = compute_step_classification(probs, labels)
        per_step[t] = {
            "mask_loss": loss_t,
            "mask_precision": cls["mask_precision"],
            "mask_recall": cls["mask_recall"],
            "mask_f1": cls["mask_f1"],
            "mask_pr_auc": cls["mask_pr_auc"],
        }

    per_transition = {}
    for t, delta in enumerate(delta_logits_steps):
        entry = {
            "refine_delta_logit_mean": delta.abs().mean().item(),
            "mask_prob_shift_mean": prob_shift_steps[t].item(),
        }
        if num_experts > 1 and t < len(gate_entropy_steps):
            entry["gate_entropy"] = gate_entropy_steps[t].item()
        per_transition[t] = entry

    return per_step, per_transition


def merge_step_metrics(accumulator, per_step, per_transition):
    for t, sm in per_step.items():
        bucket = accumulator["per_step"].setdefault(t, {k: [] for k in sm})
        for k, v in sm.items():
            bucket[k].append(v)
    for t, tr in per_transition.items():
        bucket = accumulator["per_transition"].setdefault(t, {k: [] for k in tr})
        for k, v in tr.items():
            bucket[k].append(v)


def average_step_metrics(accumulator):
    avg = {"per_step": {}, "per_transition": {}}
    for t, fields in accumulator["per_step"].items():
        avg["per_step"][t] = {k: float(np.mean(v)) for k, v in fields.items()}
    for t, fields in accumulator["per_transition"].items():
        avg["per_transition"][t] = {k: float(np.mean(v)) for k, v in fields.items()}
    return avg


def compute_refinement_diagnostics(step_metrics, num_experts=1):
    if not step_metrics or not step_metrics.get("per_step"):
        diag = {
            "mask_f1_improvement": 0.0,
            "refine_delta_logit_mean": 0.0,
            "mask_prob_shift_mean": 0.0,
        }
        if num_experts > 1:
            diag["gate_entropy"] = 0.0
        return diag
    f1_steps = [step_metrics["per_step"][t]["mask_f1"] for t in sorted(step_metrics["per_step"])]
    diag = {"mask_f1_improvement": f1_steps[-1] - f1_steps[0]}
    transitions = step_metrics.get("per_transition", {})
    if transitions:
        diag["refine_delta_logit_mean"] = float(np.mean(
            [tr["refine_delta_logit_mean"] for tr in transitions.values()]
        ))
        diag["mask_prob_shift_mean"] = float(np.mean(
            [tr["mask_prob_shift_mean"] for tr in transitions.values()]
        ))
    else:
        diag["refine_delta_logit_mean"] = 0.0
        diag["mask_prob_shift_mean"] = 0.0
    if num_experts > 1:
        diag["gate_entropy"] = compute_gate_entropy_mean(step_metrics)
    return diag


def forward_model(model, data):
    return model(data, return_refinement=True)


def joint_meanshift_loss(q_pred, mask_prob, bandwidth, joint_gt, args):
    cd_before = chamfer_distance_with_average(q_pred.unsqueeze(0), joint_gt.unsqueeze(0))
    clustered_pred = meanshift_cluster(q_pred, bandwidth, mask_prob, args)
    loss_ms = sum(
        chamfer_distance_with_average(step_pts.unsqueeze(0), joint_gt.unsqueeze(0))
        for step_pts in clustered_pred
    ) / args.meanshift_step
    cd_after = chamfer_distance_with_average(clustered_pred[-1].unsqueeze(0), joint_gt.unsqueeze(0))
    return cd_before, cd_after, loss_ms, clustered_pred


def main(args):
    global device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    args = apply_ablation_preset(args)
    set_refine_debug(args.refine_debug)
    set_seed(getattr(args, 'seed', 42))

    lowest_val_loss = 1e20
    best_tracker = BestTracker(
        lower_better={"loss"} | set(LOSS_COMPONENTS) | set(MEANSHIFT_METRICS),
        higher_better=set(REFINEMENT_METRICS),
    )

    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume and isdir(args.logdir):
        shutil.rmtree(args.logdir)
    mkdir_p(args.logdir)

    model = JOINTNET_MASKNET_MEANSHIFT(
        num_refine_steps=args.num_refine_steps,
        num_experts=args.num_experts,
        shared_refinement=SHARED_REFINEMENT,
    ).to(device)

    if args.resume and isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        args.start_epoch = checkpoint['epoch']
        lowest_val_loss = checkpoint['lowest_loss']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        resumed_tracker = checkpoint.get('best_tracker')
        if isinstance(resumed_tracker, BestTracker):
            best_tracker = resumed_tracker
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        checkpoint = None
        model.masknet.load_state_dict(torch.load(args.masknet_resume, map_location=device)['state_dict'])
        model.jointnet.load_state_dict(torch.load(args.jointnet_resume, map_location=device)['state_dict'])

    model.freeze_jointnet()
    model.verify_jointnet_frozen()
    if args.freeze_masknet:
        model.freeze_masknet()
        model.verify_masknet_frozen()

    group_index = {}
    param_groups = []
    if not args.freeze_masknet:
        group_index['masknet'] = len(param_groups)
        param_groups.append({'params': list(model.masknet.parameters()), 'lr': args.masknet_lr})
    group_index['refinement'] = len(param_groups)
    param_groups.append({'params': list(model.refinement.parameters()), 'lr': args.refine_lr})
    group_index['bandwidth'] = len(param_groups)
    param_groups.append({'params': [model.bandwidth], 'lr': args.bandwidth_lr})
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError:
            print("=> optimizer incompatible; starting fresh optimizer")

    verify_optimizer_scope(model, optimizer)

    cudnn.benchmark = True
    print('    Trainable params: %.3fM (JointNet frozen, MaskNet %s)' % (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        'frozen' if args.freeze_masknet else 'trainable'))
    print('    Refinement: steps=%d experts=%d shared=True use_bce=%s' % (
        args.num_refine_steps, args.num_experts, args.use_bce))

    train_loader = DataLoader(GraphDataset(root=args.train_folder), batch_size=args.train_batch,
                              shuffle=True, follow_batch=['joints'])
    val_loader = DataLoader(GraphDataset(root=args.val_folder), batch_size=args.test_batch,
                            shuffle=False, follow_batch=['joints'])
    test_loader = DataLoader(GraphDataset(root=args.test_folder), batch_size=args.test_batch,
                             shuffle=False, follow_batch=['joints'])

    run_name = args.ablation or f"refine_s{args.num_refine_steps}_e{args.num_experts}"
    mlflow.set_experiment("RigNet_Joint_Finetune")

    def log_run_params():
        log_args_to_mlflow(args)
        mlflow.log_param("device", str(device))
        mlflow.log_param("jointnet_frozen", True)
        mlflow.log_param("masknet_frozen", bool(args.freeze_masknet))
        mlflow.log_param("shared_refinement", True)
        mlflow.log_param("num_refine_steps", args.num_refine_steps)
        mlflow.log_param("num_experts", args.num_experts)
        mlflow.log_param("lambda_intermediate", args.lambda_intermediate)
        mlflow.log_param("bce_in_loss", bool(args.use_bce))

    if args.evaluate:
        with mlflow.start_run(run_name=f"{run_name}_eval"):
            log_run_params()
            val_loss, val_metrics, val_steps = evaluate(val_loader, model, args)
            test_loss, test_metrics, test_steps = evaluate(
                test_loader, model, args, save_result=True, best_epoch=args.start_epoch
            )
            print('val_loss {:.6f} | test_loss {:.6f}'.format(val_loss, test_loss))
            for split, loss, metrics, steps in (
                ("val", val_loss, val_metrics, val_steps),
                ("test", test_loss, test_metrics, test_steps),
            ):
                print(split + ':')
                for k, v in metrics.items():
                    print('  {}: {:.6f}'.format(k, v))
                mlflow.log_metric(f"{split}_loss", loss)
                for k in LOSS_COMPONENTS + MEANSHIFT_METRICS:
                    mlflow.log_metric(f"{split}_{k}", float(metrics[k]))
                for k in REFINEMENT_METRICS:
                    mlflow.log_metric(f"{split}_{k}", float(metrics.get(k, 0.0)))
                log_mask_step_metrics(split, steps, 0)
                log_gate_entropy_steps(split, steps, 0, args.num_experts)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)

    with mlflow.start_run(run_name=run_name):
        log_run_params()

        for epoch in range(args.start_epoch, args.epochs):
            print('\nEpoch: %d' % (epoch + 1))
            train_loss, train_metrics, train_steps = train_epoch(train_loader, model, optimizer, args)
            val_loss, val_metrics, val_steps = evaluate(val_loader, model, args)
            test_loss, test_metrics, test_steps = evaluate(test_loader, model, args)
            scheduler.step()

            print('Epoch{:d}. train_loss: {:.6f}. val_loss: {:.6f}. test_loss: {:.6f}.'.format(
                epoch + 1, train_loss, val_loss, test_loss))
            print('  base_loss  train {:.6f} / val {:.6f} / test {:.6f}'.format(
                train_metrics['base_loss'], val_metrics['base_loss'], test_metrics['base_loss']))
            print('  mask_bce   train {:.6f} / val {:.6f} / test {:.6f}'.format(
                train_metrics['bce_final'], val_metrics['bce_final'], test_metrics['bce_final']))

            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
            mlflow.log_metric("test_loss", test_loss, step=epoch + 1)
            mlflow.log_metric("lr_jointnet", 0.0, step=epoch + 1)  # frozen
            mlflow.log_metric(
                "lr_masknet",
                0.0 if args.freeze_masknet else optimizer.param_groups[group_index['masknet']]['lr'],
                step=epoch + 1,
            )
            mlflow.log_metric("lr_refinement",
                              optimizer.param_groups[group_index['refinement']]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_bandwidth",
                              optimizer.param_groups[group_index['bandwidth']]['lr'], step=epoch + 1)
            mlflow.log_metric("bandwidth", float(model.bandwidth.detach().cpu()), step=epoch + 1)

            for split, metrics in (("train", train_metrics), ("val", val_metrics), ("test", test_metrics)):
                for key in LOSS_COMPONENTS:
                    mlflow.log_metric(f"{split}_{key}", float(metrics[key]), step=epoch + 1)
                    logger.add_scalar(f"{split}/{key}", float(metrics[key]), epoch + 1)

            for k in MEANSHIFT_METRICS:
                mlflow.log_metric(f"val_{k}", float(val_metrics[k]), step=epoch + 1)
                mlflow.log_metric(f"test_{k}", float(test_metrics[k]), step=epoch + 1)
                logger.add_scalar(f"val/{k}", float(val_metrics[k]), epoch + 1)
                logger.add_scalar(f"test/{k}", float(test_metrics[k]), epoch + 1)

            logger.add_scalar("train/loss", train_loss, epoch + 1)
            logger.add_scalar("val/loss", val_loss, epoch + 1)
            logger.add_scalar("test/loss", test_loss, epoch + 1)

            # Refinement diagnostics are independent of whether BCE enters the loss.
            for split, metrics in (("train", train_metrics), ("val", val_metrics), ("test", test_metrics)):
                for key in REFINEMENT_METRICS:
                    mlflow.log_metric(f"{split}_{key}", metrics.get(key, 0.0), step=epoch + 1)
                if args.num_experts > 1 and "gate_entropy" in metrics:
                    mlflow.log_metric(f"{split}_gate_entropy", metrics["gate_entropy"], step=epoch + 1)
            for prefix, steps in (("train", train_steps), ("val", val_steps), ("test", test_steps)):
                log_mask_step_metrics(prefix, steps, epoch + 1)
                log_gate_entropy_steps(prefix, steps, epoch + 1, args.num_experts)
            if args.num_refine_steps > 0:
                log_alpha_steps(model.refinement.alpha.detach().cpu().tolist(), epoch + 1)

            best_tracker.update(val_metrics, val_loss, epoch + 1)

            is_best = val_loss < lowest_val_loss
            lowest_val_loss = min(val_loss, lowest_val_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_val_loss,
                'optimizer': optimizer.state_dict(),
                'best_tracker': best_tracker,
            }, is_best, checkpoint=args.checkpoint)

        mlflow.log_param("best_val_epoch", int(best_tracker.epochs.get("loss", 0)))
        best_tracker.log_summary("best_val")

        best_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
        print("=> loading best validation checkpoint '{}'".format(best_path))
        best_ckpt = torch.load(best_path, map_location=device)
        best_epoch = best_ckpt['epoch']
        model.load_state_dict(best_ckpt['state_dict'])

        test_loss, test_metrics, test_steps = evaluate(
            test_loader, model, args, save_result=True, best_epoch=best_epoch
        )
        print('Final test (best val epoch {}): loss {:.8f}'.format(best_epoch, test_loss))
        for k, v in test_metrics.items():
            print('  {}: {:.6f}'.format(k, v))

        mlflow.log_metric("test_loss", test_loss)
        for k in LOSS_COMPONENTS + MEANSHIFT_METRICS:
            mlflow.log_metric(f"test_{k}", float(test_metrics[k]))
        for key in REFINEMENT_METRICS:
            mlflow.log_metric(f"test_{key}", test_metrics.get(key, 0.0))
        if args.num_experts > 1 and "gate_entropy" in test_metrics:
            mlflow.log_metric("test_gate_entropy", test_metrics["gate_entropy"])
        log_mask_step_metrics("test", test_steps, best_epoch)
        log_gate_entropy_steps("test", test_steps, best_epoch, args.num_experts)


def _mask_losses(logits_steps, mask_gt, args):
    """BCE components; computed as metrics even when they do not enter the loss."""
    if args.use_bce:
        return compute_mask_bce_split(logits_steps, mask_gt)
    with torch.no_grad():
        return compute_mask_bce_split(logits_steps, mask_gt)


def train_epoch(train_loader, model, optimizer, args):
    global device
    model.train()  # frozen submodules stay in eval via JOINTNET_MASKNET_MEANSHIFT.train

    loss_meter = AverageMeter()
    base_meter = AverageMeter()
    bce_final_meter = AverageMeter()
    bce_inter_meter = AverageMeter()
    step_accumulator = {"per_step": {}, "per_transition": {}}

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = forward_model(model, data)
        q_pred = out["q_final"].detach()
        mask_prob = out["mask_prob"]
        bandwidth = out["bandwidth"]
        logits_steps = out["logits_steps"]

        base_loss = 0.0
        num_graphs = len(torch.unique(data.batch))
        for i in range(num_graphs):
            joint_gt = data.joints[data.joints_batch == i, :]
            q_i = q_pred[data.batch == i, :]
            mask_i = mask_prob[data.batch == i]
            cd_before, _, loss_ms, _ = joint_meanshift_loss(q_i, mask_i, bandwidth, joint_gt, args)
            base_loss = base_loss + cd_before + args.ms_loss_weight * loss_ms
        base_loss = base_loss / num_graphs

        mask_gt = data.mask.unsqueeze(1)
        bce_final, bce_inter = _mask_losses(logits_steps, mask_gt, args)

        loss_total = base_loss
        if args.use_bce:
            loss_total = loss_total + args.bce_loss_weight * bce_final \
                + args.lambda_intermediate * bce_inter

        with torch.no_grad():
            per_step, per_transition = collect_step_metrics(
                logits_steps, out["delta_logits_steps"], out["prob_shift_steps"],
                out["gate_entropy_steps"], mask_gt, args.num_experts,
            )
        merge_step_metrics(step_accumulator, per_step, per_transition)

        loss_total.backward()
        optimizer.step()

        loss_meter.update(loss_total.item())
        base_meter.update(base_loss.item())
        bce_final_meter.update(bce_final.item())
        bce_inter_meter.update(bce_inter.item())

    step_metrics = average_step_metrics(step_accumulator) if step_accumulator["per_step"] else None
    metrics = compute_refinement_diagnostics(step_metrics, args.num_experts)
    metrics.update({
        'base_loss': base_meter.avg,
        'bce_final': bce_final_meter.avg,
        'bce_intermediate': bce_inter_meter.avg,
    })
    return loss_meter.avg, metrics, step_metrics


def evaluate(loader, model, args, save_result=False, best_epoch=None):
    global device
    model.eval()

    loss_meter = AverageMeter()
    base_meter = AverageMeter()
    bce_final_meter = AverageMeter()
    bce_inter_meter = AverageMeter()
    cd_after_meter = AverageMeter()
    avg_shift_last_meter = AverageMeter()
    total_disp_meter = AverageMeter()
    disp_std_meter = AverageMeter()
    step_accumulator = {"per_step": {}, "per_transition": {}}

    outdir = args.checkpoint.split('/')[-1]
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = forward_model(model, data)
            q_pred = out["q_final"]
            mask_prob = out["mask_prob"]
            bandwidth = out["bandwidth"]
            logits_steps = out["logits_steps"]

            base_loss = 0.0
            num_graphs = len(torch.unique(data.batch))

            for i in range(num_graphs):
                joint_gt = data.joints[data.joints_batch == i, :]
                q_i = q_pred[data.batch == i, :]
                mask_i = mask_prob[data.batch == i]

                cd_before, cd_after, loss_ms, clustered = joint_meanshift_loss(
                    q_i, mask_i, bandwidth, joint_gt, args
                )
                base_loss = base_loss + cd_before + args.ms_loss_weight * loss_ms
                cd_after_meter.update(cd_after.item())

                if len(clustered) >= 2:
                    avg_shift_last_meter.update(
                        torch.norm(clustered[-1] - clustered[-2], dim=1).mean().item()
                    )
                total_disp_meter.update(torch.norm(clustered[-1] - q_i, dim=1).mean().item())
                disp_std_meter.update(torch.norm(clustered[-1] - q_i, dim=1).std().item())

                if save_result:
                    folder = 'results/{:s}/best_{:d}/'.format(outdir, best_epoch)
                    mkdir_p(folder)
                    output_point_cloud_ply(q_i, name=str(data.name[i].item()), output_folder=folder)
                    np.save(os.path.join(folder, '{:d}_attn.npy'.format(data.name[i].item())),
                            mask_i.cpu().numpy())
                    np.save(os.path.join(folder, '{:d}_bandwidth.npy'.format(data.name[i].item())),
                            bandwidth.cpu().numpy())

            base_loss = base_loss / num_graphs

            mask_gt = data.mask.unsqueeze(1)
            bce_final, bce_inter = compute_mask_bce_split(logits_steps, mask_gt)

            loss_total = base_loss
            if args.use_bce:
                loss_total = loss_total + args.bce_loss_weight * bce_final \
                    + args.lambda_intermediate * bce_inter

            per_step, per_transition = collect_step_metrics(
                logits_steps, out["delta_logits_steps"], out["prob_shift_steps"],
                out["gate_entropy_steps"], mask_gt, args.num_experts,
            )
            merge_step_metrics(step_accumulator, per_step, per_transition)

            loss_meter.update(loss_total.item())
            base_meter.update(base_loss.item())
            bce_final_meter.update(bce_final.item())
            bce_inter_meter.update(bce_inter.item())

    metrics = {
        'cd_after': cd_after_meter.avg,
        'avg_shift_last': avg_shift_last_meter.avg,
        'total_disp': total_disp_meter.avg,
        'disp_std': disp_std_meter.avg,
        'base_loss': base_meter.avg,
        'bce_final': bce_final_meter.avg,
        'bce_intermediate': bce_inter_meter.avg,
    }
    step_metrics = average_step_metrics(step_accumulator) if step_accumulator["per_step"] else None
    metrics.update(compute_refinement_diagnostics(step_metrics, args.num_experts))

    return loss_meter.avg, metrics, step_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iterative MaskNet finetune (JointNet frozen)')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--schedule', type=int, nargs='+', default=[50])
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--train_batch', default=1, type=int)
    parser.add_argument('--test_batch', default=1, type=int)
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str)
    parser.add_argument('--logdir', default='logs/test', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--train_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/train/', type=str)
    parser.add_argument('--val_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/val/', type=str)
    parser.add_argument('--test_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/test/', type=str)
    parser.add_argument('--masknet_lr', default=5e-5, type=float)
    parser.add_argument('--refine_lr', default=REFINE_LR_DEFAULT, type=float,
                        help='LR for the from-scratch refinement head (may exceed the backbone LRs)')
    parser.add_argument('--bandwidth_lr', default=1e-6, type=float)
    parser.add_argument('--jointnet_resume', default='checkpoints/pretrain_jointnet/model_best.pth.tar', type=str)
    parser.add_argument('--masknet_resume', default='checkpoints/pretrain_masknet/model_best.pth.tar', type=str)
    parser.add_argument('--meanshift_step', default=15, type=int)
    parser.add_argument('--step_size', default=0.3, type=float)
    parser.add_argument('--ms_loss_weight', default=2.0, type=float)
    parser.add_argument('--use_bce', action='store_true')
    parser.add_argument('--bce_loss_weight', default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)

    add_refinement_args(parser)
    parser.add_argument('--cuda', default=0, type=int, help='CUDA device index')
    args = parser.parse_args()
    print(args)
    main(args)
