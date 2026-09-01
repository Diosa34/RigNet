#-------------------------------------------------------------------------------
# Name:        run_joint_finetune.py
# Purpose:     Finetune MaskNet + iterative refinement (JointNet frozen) with mean-shift
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
sys.path.append("./")
import os
import numpy as np
import shutil
import argparse

from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import output_point_cloud_ply
from utils.log_args_to_mlflow import log_args_to_mlflow
from utils.mask_refinement_config import add_refinement_args, apply_ablation_preset

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

# Best-tracker schema: lower-is-better vs higher-is-better keys (validation only).
VAL_LOWER_BETTER = {
    "loss", "cd_after", "avg_shift_last", "total_disp", "disp_std",
    "intermediate_mask_loss",
    "cd_j2j", "cd_j2b", "cd_b2b", "tree_edit_dist",
}
VAL_HIGHER_BETTER = {
    "mask_precision", "mask_recall", "mask_f1", "mask_pr_auc",
    "mask_f1_improvement",
    "skeleton_iou", "skeleton_precision", "skeleton_recall",
}


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def pairwise_distances(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def meanshift_cluster(pts, bandwidth, weights, args):
    """Mean-shift (unchanged). Uses refined attention as per-point weights."""
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
    intermediate_loss = sum(step_losses) / len(step_losses)
    return step_losses, intermediate_loss


def compute_mask_classification_metrics(all_probs, all_labels):
    pred_bin = (all_probs > 0.5).astype(np.uint8)
    labels = all_labels.astype(np.uint8)
    return {
        "mask_precision": precision_score(labels, pred_bin, zero_division=0),
        "mask_recall": recall_score(labels, pred_bin, zero_division=0),
        "mask_f1": f1_score(labels, pred_bin, zero_division=0),
        "mask_pr_auc": average_precision_score(labels, all_probs),
    }


def collect_step_metrics(logits_steps, attention_steps, delta_logits_steps,
                         prob_shift_steps, gate_entropy_steps, mask_gt, num_experts):
    """Per-step and per-transition metrics for MLflow logging."""
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
    """Merge batch-level step metrics into running lists for epoch averaging."""
    for t, sm in per_step.items():
        bucket = accumulator["per_step"].setdefault(t, {k: [] for k in sm})
        for k, v in sm.items():
            bucket[k].append(v)
    for t, tr in per_transition.items():
        bucket = accumulator["per_transition"].setdefault(t, {k: [] for k in tr})
        for k, v in tr.items():
            bucket[k].append(v)


def average_step_metrics(accumulator):
    """Average collected per-step metrics over batches."""
    avg = {"per_step": {}, "per_transition": {}}
    for t, fields in accumulator["per_step"].items():
        avg["per_step"][t] = {k: float(np.mean(v)) for k, v in fields.items()}
    for t, fields in accumulator["per_transition"].items():
        avg["per_transition"][t] = {k: float(np.mean(v)) for k, v in fields.items()}
    return avg


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
    loss_ms = 0.0
    for step_pts in clustered_pred:
        loss_ms += chamfer_distance_with_average(step_pts.unsqueeze(0), joint_gt.unsqueeze(0))
    loss_ms /= args.meanshift_step
    cd_after = chamfer_distance_with_average(clustered_pred[-1].unsqueeze(0), joint_gt.unsqueeze(0))
    return cd_before, cd_after, loss_ms, clustered_pred


def main(args):
    global device
    args = apply_ablation_preset(args)
    lowest_val_loss = 1e20
    best_tracker = {"val": {}, "val_epochs": {}}

    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume:
        if isdir(args.logdir):
            shutil.rmtree(args.logdir)
        mkdir_p(args.logdir)

    model = JOINTNET_MASKNET_MEANSHIFT(
        num_refine_steps=args.num_refine_steps,
        num_experts=args.num_experts,
        shared_refinement=args.shared_refinement,
        refine_hidden_dim=args.refine_hidden_dim,
        gating_hidden_dim=args.gating_hidden_dim,
    )
    model.to(device)

    optimizer = torch.optim.Adam([
        {'params': model.masknet.parameters(), 'lr': args.masknet_lr},
        {'params': model.refinement.parameters(), 'lr': args.refine_lr},
        {'params': [model.bandwidth], 'lr': args.bandwidth_lr},
    ], weight_decay=args.weight_decay)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            lowest_val_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            best_tracker = checkpoint.get('best_tracker', best_tracker)
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except ValueError:
                    print("=> optimizer state incompatible (JointNet removed); starting fresh optimizer")
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        pretrained_masknet = torch.load(args.masknet_resume, map_location=device)
        pretrained_jointnet = torch.load(args.jointnet_resume, map_location=device)
        model.masknet.load_state_dict(pretrained_masknet['state_dict'])
        model.jointnet.load_state_dict(pretrained_jointnet['state_dict'])

    model.freeze_jointnet()

    cudnn.benchmark = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('    Trainable params: %.2fM (JointNet frozen)' % (trainable / 1e6))
    print('    Refinement: steps=%d experts=%d shared=%s' % (
        args.num_refine_steps, args.num_experts, args.shared_refinement))

    train_loader = DataLoader(GraphDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=True, follow_batch=['joints'])
    val_loader = DataLoader(GraphDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
    test_loader = DataLoader(GraphDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])

    if args.evaluate:
        print('\nEvaluation only')
        _, val_metrics, _ = evaluate(val_loader, model, args)
        _, test_metrics, _ = evaluate(test_loader, model, args, save_result=True, best_epoch=args.start_epoch)
        print('val:', val_metrics)
        print('test:', test_metrics)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)

    run_name = args.ablation if args.ablation else f"refine_s{args.num_refine_steps}_e{args.num_experts}"
    mlflow.set_experiment("RigNet_Joint_Finetune")
    with mlflow.start_run(run_name=run_name):
        log_args_to_mlflow(args)
        mlflow.log_param("device", str(device))
        mlflow.log_param("jointnet_frozen", True)

        for epoch in range(args.start_epoch, args.epochs):
            print('\nEpoch: %d ' % (epoch + 1))
            train_metrics = train_epoch(train_loader, model, optimizer, args)
            val_loss, val_metrics, val_step_metrics = evaluate(val_loader, model, args)
            scheduler.step()

            print('Epoch{:d}. train_loss: {:.6f}.'.format(epoch + 1, train_metrics['loss']))
            print('Epoch{:d}. val_loss: {:.6f}.'.format(epoch + 1, val_loss))

            mlflow.log_metric("train_loss", train_metrics['loss'], step=epoch + 1)
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
            mlflow.log_metric("lr_masknet", optimizer.param_groups[0]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_refinement", optimizer.param_groups[1]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_bandwidth", optimizer.param_groups[2]['lr'], step=epoch + 1)

            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v), step=epoch + 1)
                logger.add_scalar(f"val/{k}", float(v), epoch + 1)

            logger.add_scalar("train/loss", train_metrics['loss'], epoch + 1)
            logger.add_scalar("val/loss", val_loss, epoch + 1)

            if args.use_bce and val_step_metrics:
                log_step_metrics_to_mlflow("val", val_step_metrics, epoch + 1, args.num_experts)

            mlflow.log_metric("train_intermediate_mask_loss", train_metrics.get('intermediate_mask_loss', 0.0), step=epoch + 1)
            for key in ("mask_f1_improvement", "refine_delta_logit_mean", "mask_prob_shift_mean"):
                mlflow.log_metric(f"val_{key}", val_metrics.get(key, 0.0), step=epoch + 1)

            _update_best_tracker(best_tracker, val_metrics, val_loss, epoch + 1)

            is_best = val_loss < lowest_val_loss
            lowest_val_loss = min(val_loss, lowest_val_loss)
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lowest_loss': lowest_val_loss,
                    'optimizer': optimizer.state_dict(),
                    'best_tracker': best_tracker,
                },
                is_best,
                checkpoint=args.checkpoint,
            )

        _log_best_val_summary(best_tracker)

        best_model_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
        print("=> loading best validation checkpoint '{}'".format(best_model_path))
        checkpoint = torch.load(best_model_path, map_location=device)
        best_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (best val epoch {})".format(best_epoch))

        test_loss, test_metrics, test_step_metrics = evaluate(
            test_loader, model, args, save_result=True, best_epoch=best_epoch
        )
        print('Final test (best val epoch {}): test_loss {:.8f}'.format(best_epoch, test_loss))
        for k, v in test_metrics.items():
            print('  {}: {:.6f}'.format(k, v))

        mlflow.log_param("best_val_epoch", int(best_epoch))
        mlflow.log_metric("final_test_loss", test_loss)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"final_test_{k}", float(v))
        if args.use_bce and test_step_metrics:
            log_step_metrics_to_mlflow("final_test", test_step_metrics, best_epoch, args.num_experts)


def _update_best_tracker(tracker, metrics, loss, epoch):
    """Update best metrics from validation only."""
    store = tracker.setdefault("val", {})
    epoch_store = tracker.setdefault("val_epochs", {})

    if "loss" not in store or loss < store["loss"]:
        store["loss"] = loss
        epoch_store["loss"] = epoch

    for key, val in metrics.items():
        if key in VAL_LOWER_BETTER:
            if key not in store or val < store[key]:
                store[key] = val
                epoch_store[key] = epoch
        elif key in VAL_HIGHER_BETTER:
            if key not in store or val > store[key]:
                store[key] = val
                epoch_store[key] = epoch


def _log_best_val_summary(best_tracker):
    """Persist best validation summary to MLflow at end of training."""
    epoch_store = best_tracker.get("val_epochs", {})
    best_epoch = epoch_store.get("loss", 0)
    mlflow.log_param("best_epoch", int(best_epoch))
    for key, val in best_tracker.get("val", {}).items():
        mlflow.log_metric(f"best_val_{key}", float(val))
        if key in epoch_store:
            mlflow.log_param(f"best_val_{key}_epoch", int(epoch_store[key]))


def train_epoch(train_loader, model, optimizer, args):
    global device
    model.train()
    model.jointnet.eval()

    loss_meter = AverageMeter()
    intermediate_meter = AverageMeter()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = forward_model(model, data)
        q_pred = out["q_final"]
        logits_steps = out["logits_steps"]
        mask_prob = out["mask_prob"]
        bandwidth = out["bandwidth"]

        loss_total = 0.0
        num_graphs = len(torch.unique(data.batch))
        for i in range(num_graphs):
            joint_gt = data.joints[data.joints_batch == i, :]
            q_pred_i = q_pred[data.batch == i, :]
            mask_prob_i = mask_prob[data.batch == i]
            cd_before, _, loss_ms, _ = joint_meanshift_loss(q_pred_i, mask_prob_i, bandwidth, joint_gt, args)
            loss_total += cd_before + args.ms_loss_weight * loss_ms

        loss_total /= num_graphs

        if args.use_bce:
            mask_gt = data.mask.unsqueeze(1)
            step_losses, intermediate_loss = compute_mask_bce(logits_steps, mask_gt)
            final_mask_loss = step_losses[-1]
            loss_total = loss_total + args.bce_loss_weight * final_mask_loss
            loss_total = loss_total + args.lambda_intermediate * intermediate_loss
            intermediate_meter.update(intermediate_loss.item())

        loss_total.backward()
        optimizer.step()
        loss_meter.update(loss_total.item())

    return {
        "loss": loss_meter.avg,
        "intermediate_mask_loss": intermediate_meter.avg,
    }


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
    all_probs_final = []
    all_probs_step0 = []
    all_labels = []

    outdir = args.checkpoint.split('/')[-1]
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = forward_model(model, data)
            q_pred = out["q_final"]
            logits_steps = out["logits_steps"]
            mask_prob = out["mask_prob"]
            bandwidth = out["bandwidth"]

            loss_total = 0.0
            num_graphs = len(torch.unique(data.batch))

            for i in range(num_graphs):
                joint_gt = data.joints[data.joints_batch == i, :]
                q_pred_i = q_pred[data.batch == i, :]
                mask_prob_i = mask_prob[data.batch == i]

                cd_before, cd_after, loss_ms, clustered_pred = joint_meanshift_loss(
                    q_pred_i, mask_prob_i, bandwidth, joint_gt, args
                )
                loss_total += cd_before + args.ms_loss_weight * loss_ms
                cd_after_meter.update(cd_after.item())

                if len(clustered_pred) >= 2:
                    shift_last = torch.norm(clustered_pred[-1] - clustered_pred[-2], dim=1).mean().item()
                else:
                    shift_last = 0.0
                avg_shift_last_meter.update(shift_last)

                y_pred_final = clustered_pred[-1]
                total_disp = torch.norm(y_pred_final - q_pred_i, dim=1).mean().item()
                total_disp_meter.update(total_disp)
                disp_std_meter.update(torch.norm(y_pred_final - q_pred_i, dim=1).std().item())

                if save_result:
                    output_folder = 'results/{:s}/best_{:d}/'.format(outdir, best_epoch)
                    if not os.path.exists(output_folder):
                        mkdir_p(output_folder)
                    output_point_cloud_ply(q_pred_i, name=str(data.name[i].item()), output_folder=output_folder)
                    np.save(os.path.join(output_folder, '{:d}_attn.npy'.format(data.name[i].item())),
                            mask_prob_i.data.to("cpu").numpy())
                    np.save(os.path.join(output_folder, '{:d}_bandwidth.npy'.format(data.name[i].item())),
                            bandwidth.data.to("cpu").numpy())

            loss_total /= num_graphs

            if args.use_bce:
                mask_gt = data.mask.unsqueeze(1)
                step_losses, intermediate_loss = compute_mask_bce(logits_steps, mask_gt)
                loss_total = loss_total + args.bce_loss_weight * step_losses[-1]
                loss_total = loss_total + args.lambda_intermediate * intermediate_loss
                intermediate_meter.update(intermediate_loss.item())

                per_step, per_transition = collect_step_metrics(
                    logits_steps,
                    out["attention_steps"],
                    out["delta_logits_steps"],
                    out["prob_shift_steps"],
                    out["gate_entropy_steps"],
                    mask_gt,
                    args.num_experts,
                )
                merge_step_metrics(step_accumulator, per_step, per_transition)

                all_probs_final.append(torch.sigmoid(logits_steps[-1]).detach().cpu().numpy().reshape(-1))
                all_probs_step0.append(torch.sigmoid(logits_steps[0]).detach().cpu().numpy().reshape(-1))
                all_labels.append(mask_gt.detach().cpu().numpy().reshape(-1))

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
        final_cls = compute_mask_classification_metrics(
            np.concatenate(all_probs_final), np.concatenate(all_labels)
        )
        step0_cls = compute_mask_classification_metrics(
            np.concatenate(all_probs_step0), np.concatenate(all_labels)
        )
        metrics.update(final_cls)
        metrics["mask_f1_step0"] = step0_cls["mask_f1"]

        if step_metrics and step_metrics["per_transition"]:
            metrics["refine_delta_logit_mean"] = float(np.mean([
                tr["refine_delta_logit_mean"] for tr in step_metrics["per_transition"].values()
            ]))
            metrics["mask_prob_shift_mean"] = float(np.mean([
                tr["mask_prob_shift_mean"] for tr in step_metrics["per_transition"].values()
            ]))
            f1_steps = [step_metrics["per_step"][t]["mask_f1"] for t in sorted(step_metrics["per_step"])]
            metrics["mask_f1_improvement"] = f1_steps[-1] - f1_steps[0]
        else:
            metrics["mask_f1_improvement"] = final_cls["mask_f1"] - step0_cls["mask_f1"]
            metrics["refine_delta_logit_mean"] = 0.0
            metrics["mask_prob_shift_mean"] = 0.0

    return loss_meter.avg, metrics, step_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask refinement finetune (JointNet frozen)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--epochs', default=100, type=int, metavar='N')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50])
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--train_batch', default=1, type=int, metavar='N')
    parser.add_argument('--test_batch', default=1, type=int, metavar='N')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, metavar='PATH')
    parser.add_argument('--logdir', default='logs/test', type=str, metavar='LOG')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--train_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/train/', type=str)
    parser.add_argument('--val_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/val/', type=str)
    parser.add_argument('--test_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/test/', type=str)
    parser.add_argument('--jointnet_lr', default=5e-5, type=float, help='unused (JointNet frozen)')
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
