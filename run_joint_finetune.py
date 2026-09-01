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
from utils.mask_refinement_config import add_refinement_args, apply_ablation_preset, log_experiment_config

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.GCN import JOINTNET_MASKNET_MEANSHIFT
from datasets.skeleton_dataset import GraphDataset
from models.supplemental_layers.pytorch_chamfer_dist import chamfer_distance_with_average

import mlflow

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Validation-only best tracking (test never participates in model selection).
VAL_BEST_MIN = {"loss", "cd_after", "avg_shift_last", "total_disp", "disp_std", "intermediate_mask_loss",
                "cd_j2j", "cd_j2b", "cd_b2b", "tree_edit_dist"}
VAL_BEST_MAX = {"mask_precision", "mask_recall", "mask_f1", "mask_pr_auc", "mask_f1_improvement",
                "skeleton_iou", "skeleton_precision", "skeleton_recall"}


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


def verify_optimizer_excludes_jointnet(model, optimizer):
    optim_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    for p in model.jointnet.parameters():
        assert id(p) not in optim_ids, "JointNet parameters must not be in optimizer"
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(id(p) in optim_ids for p in trainable), "All trainable params must be in optimizer"


def pairwise_distances(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def meanshift_cluster(pts, bandwidth, weights, args):
    """Existing mean-shift (unchanged)."""
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


def compute_mask_bce(logits_steps, mask_gt):
    step_losses = [
        F.binary_cross_entropy_with_logits(logits, mask_gt.float(), reduction='mean')
        for logits in logits_steps
    ]
    return step_losses, sum(step_losses) / len(step_losses)


def compute_mask_classification_metrics(all_probs, all_labels):
    pred_bin = (all_probs > 0.5).astype(np.uint8)
    labels = all_labels.astype(np.uint8)
    return {
        "mask_precision": precision_score(labels, pred_bin, zero_division=0),
        "mask_recall": recall_score(labels, pred_bin, zero_division=0),
        "mask_f1": f1_score(labels, pred_bin, zero_division=0),
        "mask_pr_auc": average_precision_score(labels, all_probs),
    }


def collect_step_metrics(logits_steps, delta_logits_steps, prob_shift_steps,
                         gate_entropy_steps, mask_gt, num_experts):
    """Classification metrics at every intermediate step t=0..T."""
    per_step = {}
    labels = mask_gt.detach().cpu().numpy().reshape(-1).astype(np.uint8)

    for t, logits in enumerate(logits_steps):
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        loss_t = F.binary_cross_entropy_with_logits(logits, mask_gt.float(), reduction='mean').item()
        cls = compute_mask_classification_metrics(probs, labels)
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


def compute_aggregate_diagnostics(step_metrics):
    """Three required aggregate diagnostics from per-step classification metrics."""
    if not step_metrics or not step_metrics.get("per_step"):
        return {"mask_f1_improvement": 0.0, "refine_delta_logit_mean": 0.0, "mask_prob_shift_mean": 0.0}

    f1_steps = [step_metrics["per_step"][t]["mask_f1"] for t in sorted(step_metrics["per_step"])]
    diag = {"mask_f1_improvement": f1_steps[-1] - f1_steps[0]}

    transitions = step_metrics.get("per_transition", {})
    if transitions:
        diag["refine_delta_logit_mean"] = float(np.mean([tr["refine_delta_logit_mean"] for tr in transitions.values()]))
        diag["mask_prob_shift_mean"] = float(np.mean([tr["mask_prob_shift_mean"] for tr in transitions.values()]))
    else:
        diag["refine_delta_logit_mean"] = 0.0
        diag["mask_prob_shift_mean"] = 0.0
    return diag


def log_step_metrics_to_mlflow(prefix, step_metrics, epoch, num_experts):
    for t, sm in step_metrics["per_step"].items():
        mlflow.log_metric(f"{prefix}_mask_loss_step{t}", sm["mask_loss"], step=epoch)
        mlflow.log_metric(f"{prefix}_mask_precision_step{t}", sm["mask_precision"], step=epoch)
        mlflow.log_metric(f"{prefix}_mask_recall_step{t}", sm["mask_recall"], step=epoch)
        mlflow.log_metric(f"{prefix}_mask_f1_step{t}", sm["mask_f1"], step=epoch)
        mlflow.log_metric(f"{prefix}_mask_pr_auc_step{t}", sm["mask_pr_auc"], step=epoch)

    for t, tr in step_metrics["per_transition"].items():
        mlflow.log_metric(f"{prefix}_refine_delta_logit_mean_step{t}", tr["refine_delta_logit_mean"], step=epoch)
        mlflow.log_metric(f"{prefix}_mask_prob_shift_mean_step{t}", tr["mask_prob_shift_mean"], step=epoch)
        if num_experts > 1 and "gate_entropy" in tr:
            mlflow.log_metric(f"{prefix}_gate_entropy_step{t}", tr["gate_entropy"], step=epoch)


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
    args = apply_ablation_preset(args)
    set_seed(args.seed)

    lowest_val_loss = 1e20
    best_tracker = {"val": {}, "val_epochs": {}}

    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume and isdir(args.logdir):
        shutil.rmtree(args.logdir)
    mkdir_p(args.logdir)

    model = JOINTNET_MASKNET_MEANSHIFT(
        num_refine_steps=args.num_refine_steps,
        num_experts=args.num_experts,
        shared_refinement=args.shared_refinement,
        refine_hidden_dim=args.refine_hidden_dim,
        gating_hidden_dim=args.gating_hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.masknet.parameters(), 'lr': args.masknet_lr},
        {'params': model.refinement.parameters(), 'lr': args.refine_lr},
        {'params': [model.bandwidth], 'lr': args.bandwidth_lr},
    ], weight_decay=args.weight_decay)

    if args.resume and isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        args.start_epoch = checkpoint['epoch']
        lowest_val_loss = checkpoint['lowest_loss']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        best_tracker = checkpoint.get('best_tracker', best_tracker)
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError:
            print("=> optimizer incompatible; starting fresh optimizer")
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        model.masknet.load_state_dict(torch.load(args.masknet_resume, map_location=device)['state_dict'])
        model.jointnet.load_state_dict(torch.load(args.jointnet_resume, map_location=device)['state_dict'])

    model.freeze_jointnet()
    model.verify_jointnet_frozen()
    verify_optimizer_excludes_jointnet(model, optimizer)

    cudnn.benchmark = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('    Trainable params: %.2fM (JointNet frozen)' % (trainable / 1e6))
    print('    Refinement: steps=%d experts=%d shared=%s seed=%d' % (
        args.num_refine_steps, args.num_experts, args.shared_refinement, args.seed))

    train_loader = DataLoader(GraphDataset(root=args.train_folder), batch_size=args.train_batch,
                              shuffle=True, follow_batch=['joints'])
    val_loader = DataLoader(GraphDataset(root=args.val_folder), batch_size=args.test_batch,
                            shuffle=False, follow_batch=['joints'])
    test_loader = DataLoader(GraphDataset(root=args.test_folder), batch_size=args.test_batch,
                             shuffle=False, follow_batch=['joints'])

    if args.evaluate:
        _, val_metrics, val_steps = evaluate(val_loader, model, args)
        _, test_metrics, test_steps = evaluate(test_loader, model, args, save_result=True, best_epoch=args.start_epoch)
        print('val:', val_metrics)
        print('test:', test_metrics)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)
    run_name = args.ablation or f"refine_s{args.num_refine_steps}_e{args.num_experts}"

    mlflow.set_experiment("RigNet_Joint_Finetune")
    with mlflow.start_run(run_name=run_name):
        log_args_to_mlflow(args)
        log_experiment_config(args)
        mlflow.log_param("device", str(device))

        for epoch in range(args.start_epoch, args.epochs):
            print('\nEpoch: %d' % (epoch + 1))
            train_loss, train_metrics, train_steps = train_epoch(train_loader, model, optimizer, args)
            val_loss, val_metrics, val_steps = evaluate(val_loader, model, args)
            scheduler.step()

            print('Epoch{:d}. train_loss: {:.6f}. val_loss: {:.6f}.'.format(epoch + 1, train_loss, val_loss))

            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
            mlflow.log_metric("lr_masknet", optimizer.param_groups[0]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_refinement", optimizer.param_groups[1]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_bandwidth", optimizer.param_groups[2]['lr'], step=epoch + 1)

            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v), step=epoch + 1)
                logger.add_scalar(f"val/{k}", float(v), epoch + 1)

            logger.add_scalar("train/loss", train_loss, epoch + 1)
            logger.add_scalar("val/loss", val_loss, epoch + 1)

            if args.use_bce:
                mlflow.log_metric("train_intermediate_mask_loss",
                                  train_metrics.get("intermediate_mask_loss", 0.0), step=epoch + 1)
                if train_steps:
                    log_step_metrics_to_mlflow("train", train_steps, epoch + 1, args.num_experts)
                if val_steps:
                    log_step_metrics_to_mlflow("val", val_steps, epoch + 1, args.num_experts)
                for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                    for key in ("mask_f1_improvement", "refine_delta_logit_mean", "mask_prob_shift_mean"):
                        mlflow.log_metric(f"{split}_{key}", metrics.get(key, 0.0), step=epoch + 1)

            _update_best_tracker(best_tracker, val_metrics, val_loss, epoch + 1)

            is_best = val_loss < lowest_val_loss
            lowest_val_loss = min(val_loss, lowest_val_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_val_loss,
                'optimizer': optimizer.state_dict(),
                'best_tracker': best_tracker,
            }, is_best, checkpoint=args.checkpoint)

        _log_best_val_summary(best_tracker)

        best_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
        print("=> loading best validation checkpoint '{}'".format(best_path))
        best_ckpt = torch.load(best_path, map_location=device)
        best_epoch = best_ckpt['epoch']
        model.load_state_dict(best_ckpt['state_dict'])
        mlflow.log_artifact(best_path, artifact_path="checkpoints")

        test_loss, test_metrics, test_steps = evaluate(
            test_loader, model, args, save_result=True, best_epoch=best_epoch
        )
        print('Final test (best val epoch {}): loss {:.8f}'.format(best_epoch, test_loss))
        for k, v in test_metrics.items():
            print('  {}: {:.6f}'.format(k, v))

        mlflow.log_param("best_val_epoch", int(best_epoch))
        mlflow.log_metric("final_test_loss", test_loss)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"final_test_{k}", float(v))
        if args.use_bce and test_steps:
            log_step_metrics_to_mlflow("final_test", test_steps, best_epoch, args.num_experts)


def _update_best_tracker(tracker, metrics, loss, epoch):
    store = tracker.setdefault("val", {})
    epochs = tracker.setdefault("val_epochs", {})

    if "loss" not in store or loss < store["loss"]:
        store["loss"] = loss
        epochs["loss"] = epoch

    for key, val in metrics.items():
        if key in VAL_BEST_MIN:
            if key not in store or val < store[key]:
                store[key] = val
                epochs[key] = epoch
        elif key in VAL_BEST_MAX:
            if key not in store or val > store[key]:
                store[key] = val
                epochs[key] = epoch


def _log_best_val_summary(best_tracker):
    """Log best validation metrics only (no best_test_*)."""
    store = best_tracker.get("val", {})
    epochs = best_tracker.get("val_epochs", {})

    best_epoch = epochs.get("loss", 0)
    mlflow.log_param("best_val_epoch", int(best_epoch))

    mapping = {
        "loss": "best_val_loss",
        "mask_precision": "best_val_precision",
        "mask_recall": "best_val_recall",
        "mask_f1": "best_val_f1",
        "mask_pr_auc": "best_val_pr_auc",
    }
    for src, dst in mapping.items():
        if src in store:
            mlflow.log_metric(dst, float(store[src]))
            if src in epochs:
                mlflow.log_param(f"{dst}_epoch", int(epochs[src]))

    for key, val in store.items():
        if key not in mapping:
            mlflow.log_metric(f"best_val_{key}", float(val))
            if key in epochs:
                mlflow.log_param(f"best_val_{key}_epoch", int(epochs[key]))


def train_epoch(train_loader, model, optimizer, args):
    global device
    model.train()
    model.jointnet.eval()

    loss_meter = AverageMeter()
    intermediate_meter = AverageMeter()
    step_accumulator = {"per_step": {}, "per_transition": {}}

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = forward_model(model, data)
        q_pred = out["q_final"].detach()  # JointNet frozen: no grad through geometry
        mask_prob = out["mask_prob"]
        bandwidth = out["bandwidth"]
        logits_steps = out["logits_steps"]

        loss_total = 0.0
        num_graphs = len(torch.unique(data.batch))
        for i in range(num_graphs):
            joint_gt = data.joints[data.joints_batch == i, :]
            q_i = q_pred[data.batch == i, :]
            mask_i = mask_prob[data.batch == i]
            _, _, loss_ms, _ = joint_meanshift_loss(q_i, mask_i, bandwidth, joint_gt, args)
            loss_total += args.ms_loss_weight * loss_ms
        loss_total /= num_graphs

        if args.use_bce:
            mask_gt = data.mask.unsqueeze(1)
            step_losses, intermediate_loss = compute_mask_bce(logits_steps, mask_gt)
            loss_total = loss_total + args.bce_loss_weight * step_losses[-1]
            loss_total = loss_total + args.lambda_intermediate * intermediate_loss
            intermediate_meter.update(intermediate_loss.item())

            per_step, per_transition = collect_step_metrics(
                logits_steps, out["delta_logits_steps"], out["prob_shift_steps"],
                out["gate_entropy_steps"], mask_gt, args.num_experts,
            )
            merge_step_metrics(step_accumulator, per_step, per_transition)

        loss_total.backward()
        optimizer.step()
        loss_meter.update(loss_total.item())

    step_metrics = average_step_metrics(step_accumulator) if step_accumulator["per_step"] else None
    metrics = {"intermediate_mask_loss": intermediate_meter.avg}
    metrics.update(compute_aggregate_diagnostics(step_metrics))
    return loss_meter.avg, metrics, step_metrics


def evaluate(loader, model, args, save_result=False, best_epoch=None):
    global device
    model.eval()

    loss_meter = AverageMeter()
    cd_after_meter = AverageMeter()
    avg_shift_last_meter = AverageMeter()
    total_disp_meter = AverageMeter()
    disp_std_meter = AverageMeter()
    intermediate_meter = AverageMeter()
    step_accumulator = {"per_step": {}, "per_transition": {}}
    all_probs_final, all_probs_step0, all_labels = [], [], []

    outdir = args.checkpoint.split('/')[-1]
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = forward_model(model, data)
            q_pred = out["q_final"]
            mask_prob = out["mask_prob"]
            bandwidth = out["bandwidth"]
            logits_steps = out["logits_steps"]

            loss_total = 0.0
            num_graphs = len(torch.unique(data.batch))

            for i in range(num_graphs):
                joint_gt = data.joints[data.joints_batch == i, :]
                q_i = q_pred[data.batch == i, :]
                mask_i = mask_prob[data.batch == i]

                cd_before, cd_after, loss_ms, clustered = joint_meanshift_loss(
                    q_i, mask_i, bandwidth, joint_gt, args
                )
                loss_total += cd_before + args.ms_loss_weight * loss_ms
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

            loss_total /= num_graphs

            if args.use_bce:
                mask_gt = data.mask.unsqueeze(1)
                step_losses, intermediate_loss = compute_mask_bce(logits_steps, mask_gt)
                loss_total += args.bce_loss_weight * step_losses[-1]
                loss_total += args.lambda_intermediate * intermediate_loss
                intermediate_meter.update(intermediate_loss.item())

                per_step, per_transition = collect_step_metrics(
                    logits_steps, out["delta_logits_steps"], out["prob_shift_steps"],
                    out["gate_entropy_steps"], mask_gt, args.num_experts,
                )
                merge_step_metrics(step_accumulator, per_step, per_transition)
                all_probs_final.append(torch.sigmoid(logits_steps[-1]).cpu().numpy().reshape(-1))
                all_probs_step0.append(torch.sigmoid(logits_steps[0]).cpu().numpy().reshape(-1))
                all_labels.append(mask_gt.cpu().numpy().reshape(-1))

            loss_meter.update(loss_total.item())

    metrics = {
        'cd_after': cd_after_meter.avg,
        'avg_shift_last': avg_shift_last_meter.avg,
        'total_disp': total_disp_meter.avg,
        'disp_std': disp_std_meter.avg,
        'intermediate_mask_loss': intermediate_meter.avg,
    }
    step_metrics = average_step_metrics(step_accumulator) if step_accumulator["per_step"] else None

    if args.use_bce and all_labels:
        metrics.update(compute_mask_classification_metrics(
            np.concatenate(all_probs_final), np.concatenate(all_labels)
        ))
        metrics.update(compute_aggregate_diagnostics(step_metrics))

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
    parser.add_argument('--bandwidth_lr', default=1e-6, type=float)
    parser.add_argument('--jointnet_resume', default='checkpoints/pretrain_jointnet/model_best.pth.tar', type=str)
    parser.add_argument('--masknet_resume', default='checkpoints/pretrain_masknet/model_best.pth.tar', type=str)
    parser.add_argument('--meanshift_step', default=15, type=int)
    parser.add_argument('--step_size', default=0.3, type=float)
    parser.add_argument('--ms_loss_weight', default=2.0, type=float)
    parser.add_argument('--use_bce', action='store_true')
    parser.add_argument('--bce_loss_weight', default=0.1, type=float)

    add_refinement_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
