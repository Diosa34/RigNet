#-------------------------------------------------------------------------------
# Name:        run_joint_finetune.py
# Purpose:     Finetuning JointNet + MaskNet with iterative mask refinement and mean-shift
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
import mlflow.pytorch

from sklearn.metrics import f1_score, average_precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    """
    Mean-shift written in PyTorch. Uses refined attention as per-point weights.
    JointNet displacement and this algorithm are unchanged from the original pipeline.
    """
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
    """BCE on every refinement step; returns per-step losses and their mean."""
    step_losses = [
        F.binary_cross_entropy_with_logits(logits, mask_gt.float(), reduction='mean')
        for logits in logits_steps
    ]
    intermediate_loss = sum(step_losses) / len(step_losses)
    return step_losses, intermediate_loss


def compute_refinement_diagnostics(out):
    """Three hypothesis-specific diagnostics for iterative mask reasoning."""
    delta_logits = out["delta_logits_steps"]
    delta_mean = sum(d.abs().mean().item() for d in delta_logits) / max(len(delta_logits), 1)

    logits_steps = out["logits_steps"]
    attn_steps = out["attention_steps"]
    prob_shift = (attn_steps[-1] - attn_steps[0]).abs().mean().item()

    gate_entropy = 0.0
    if out["gate_entropy_steps"]:
        gate_entropy = sum(e.item() for e in out["gate_entropy_steps"]) / len(out["gate_entropy_steps"])

    return {
        "refine_delta_logit_mean": delta_mean,
        "mask_prob_shift_mean": prob_shift,
        "gate_entropy_mean": gate_entropy,
    }


def compute_mask_classification_metrics(all_probs, all_labels):
    pred_bin = (all_probs > 0.5).astype(np.uint8)
    return {
        "mask_f1": f1_score(all_labels, pred_bin, zero_division=0),
        "mask_pr_auc": average_precision_score(all_labels, all_probs),
    }


def forward_model(model, data):
    return model(data, return_refinement=True)


def joint_meanshift_loss(y_pred, mask_prob, bandwidth, joint_gt, args):
    cd_before = chamfer_distance_with_average(y_pred.unsqueeze(0), joint_gt.unsqueeze(0))
    clustered_pred = meanshift_cluster(y_pred, bandwidth, mask_prob, args)
    loss_ms = 0.0
    for step_pts in clustered_pred:
        loss_ms += chamfer_distance_with_average(step_pts.unsqueeze(0), joint_gt.unsqueeze(0))
    loss_ms /= args.meanshift_step
    cd_after = chamfer_distance_with_average(clustered_pred[-1].unsqueeze(0), joint_gt.unsqueeze(0))
    return cd_before, cd_after, loss_ms, clustered_pred


def main(args):
    global device
    args = apply_ablation_preset(args)
    lowest_loss = 1e20
    best_tracker = {"val": {}, "test": {}, "val_epochs": {}, "test_epochs": {}}

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
        {'params': model.jointnet.parameters(), 'lr': args.jointnet_lr},
        {'params': model.masknet.parameters(), 'lr': args.masknet_lr},
        {'params': model.refinement.parameters(), 'lr': args.refine_lr},
        {'params': [model.bandwidth], 'lr': args.bandwidth_lr},
    ], weight_decay=args.weight_decay)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_tracker = checkpoint.get('best_tracker', best_tracker)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        pretrained_masknet = torch.load(args.masknet_resume, map_location=device)
        pretrained_jointnet = torch.load(args.jointnet_resume, map_location=device)
        model.masknet.load_state_dict(pretrained_masknet['state_dict'])
        model.jointnet.load_state_dict(pretrained_jointnet['state_dict'])

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('    Refinement: steps=%d experts=%d shared=%s' % (
        args.num_refine_steps, args.num_experts, args.shared_refinement))

    train_loader = DataLoader(GraphDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=True, follow_batch=['joints'])
    val_loader = DataLoader(GraphDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
    test_loader = DataLoader(GraphDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])

    if args.evaluate:
        print('\nEvaluation only')
        _, val_metrics = evaluate(val_loader, model, args)
        _, test_metrics = evaluate(test_loader, model, args, save_result=True, best_epoch=args.start_epoch)
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

        for epoch in range(args.start_epoch, args.epochs):
            print('\nEpoch: %d ' % (epoch + 1))
            train_metrics = train_epoch(train_loader, model, optimizer, args)
            val_loss, val_metrics = evaluate(val_loader, model, args)
            test_loss, test_metrics = evaluate(test_loader, model, args)
            scheduler.step()

            print('Epoch{:d}. train_loss: {:.6f}.'.format(epoch + 1, train_metrics['loss']))
            print('Epoch{:d}. val_loss: {:.6f}.'.format(epoch + 1, val_loss))
            print('Epoch{:d}. test_loss: {:.6f}.'.format(epoch + 1, test_loss))

            # --- existing metrics (unchanged keys) ---
            mlflow.log_metric("train_loss", train_metrics['loss'], step=epoch + 1)
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
            mlflow.log_metric("test_loss", test_loss, step=epoch + 1)
            mlflow.log_metric("lr_jointnet", optimizer.param_groups[0]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_masknet", optimizer.param_groups[1]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_refinement", optimizer.param_groups[2]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_bandwidth", optimizer.param_groups[3]['lr'], step=epoch + 1)

            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v), step=epoch + 1)
                logger.add_scalar(f"val/{k}", float(v), epoch + 1)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", float(v), step=epoch + 1)
                logger.add_scalar(f"test/{k}", float(v), epoch + 1)

            logger.add_scalar("train/loss", train_metrics['loss'], epoch + 1)
            logger.add_scalar("val/loss", val_loss, epoch + 1)
            logger.add_scalar("test/loss", test_loss, epoch + 1)

            # --- 3 hypothesis-specific metrics ---
            mlflow.log_metric("train_intermediate_mask_loss", train_metrics['intermediate_mask_loss'], step=epoch + 1)
            mlflow.log_metric("val_mask_f1_improvement", val_metrics.get('mask_f1_improvement', 0.0), step=epoch + 1)
            mlflow.log_metric("val_refine_delta_logit_mean", val_metrics.get('refine_delta_logit_mean', 0.0), step=epoch + 1)

            mlflow.log_metric("test_intermediate_mask_loss", test_metrics.get('intermediate_mask_loss', 0.0), step=epoch + 1)
            mlflow.log_metric("test_mask_f1_improvement", test_metrics.get('mask_f1_improvement', 0.0), step=epoch + 1)
            mlflow.log_metric("test_refine_delta_logit_mean", test_metrics.get('refine_delta_logit_mean', 0.0), step=epoch + 1)

            # track best metrics (lower is better for loss/cd, higher for f1/pr_auc)
            _update_best_tracker(best_tracker, "val", val_metrics, val_loss, epoch + 1)
            _update_best_tracker(best_tracker, "test", test_metrics, test_loss, epoch + 1)

            is_best = val_loss < lowest_loss
            lowest_loss = min(val_loss, lowest_loss)
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lowest_loss': lowest_loss,
                    'optimizer': optimizer.state_dict(),
                    'best_tracker': best_tracker,
                },
                is_best,
                checkpoint=args.checkpoint,
            )

        _log_best_summary(best_tracker)

        best_model_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
        print("=> loading checkpoint '{}'".format(best_model_path))
        checkpoint = torch.load(best_model_path, map_location=device)
        best_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(best_model_path, best_epoch))
        test_loss, test_metrics = evaluate(test_loader, model, args, save_result=True, best_epoch=best_epoch)
        print('Best epoch:\n test_loss {:.8f}'.format(test_loss))
        for k, v in test_metrics.items():
            print('  {}: {:.6f}'.format(k, v))


def _update_best_tracker(tracker, split, metrics, loss, epoch):
    """Track best achieved values per split."""
    store = tracker.setdefault(split, {})
    epoch_store = tracker.setdefault(f"{split}_epochs", {})

    lower_better = {
        "loss", "cd_after", "avg_shift_last", "total_disp", "disp_std",
        "intermediate_mask_loss",
    }
    higher_better = {
        "mask_f1", "mask_pr_auc", "mask_f1_improvement", "mask_f1_step0",
    }

    if "loss" not in store or loss < store["loss"]:
        store["loss"] = loss
        epoch_store["loss"] = epoch

    for key, val in metrics.items():
        if key in lower_better:
            if key not in store or val < store[key]:
                store[key] = val
                epoch_store[key] = epoch
        elif key in higher_better:
            if key not in store or val > store[key]:
                store[key] = val
                epoch_store[key] = epoch


def _log_best_summary(best_tracker):
    """Persist best validation/test summary metrics to MLflow at end of run."""
    best_epoch = best_tracker.get("best_epoch", 0)
    for split in ("val", "test"):
        epoch_store = best_tracker.get(f"{split}_epochs", {})
        for key, val in best_tracker.get(split, {}).items():
            mlflow.log_metric(f"best_{split}_{key}", float(val))
            if key in epoch_store:
                mlflow.log_param(f"best_{split}_{key}_epoch", int(epoch_store[key]))
                if split == "val" and key == "loss":
                    best_epoch = epoch_store[key]
    mlflow.log_param("best_epoch", int(best_epoch))


def train_epoch(train_loader, model, optimizer, args):
    global device
    model.train()
    loss_meter = AverageMeter()
    intermediate_meter = AverageMeter()
    refine_delta_meter = AverageMeter()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = forward_model(model, data)
        y_pred = out["x_offset"] + data.pos
        logits_steps = out["logits_steps"]
        mask_prob = out["mask_prob"]
        bandwidth = out["bandwidth"]

        loss_total = 0.0
        num_graphs = len(torch.unique(data.batch))
        for i in range(num_graphs):
            joint_gt = data.joints[data.joints_batch == i, :]
            y_pred_i = y_pred[data.batch == i, :]
            mask_prob_i = mask_prob[data.batch == i]
            cd_before, _, loss_ms, _ = joint_meanshift_loss(y_pred_i, mask_prob_i, bandwidth, joint_gt, args)
            loss_total += cd_before + args.ms_loss_weight * loss_ms

        loss_total /= num_graphs

        if args.use_bce:
            mask_gt = data.mask.unsqueeze(1)
            step_losses, intermediate_loss = compute_mask_bce(logits_steps, mask_gt)
            final_mask_loss = step_losses[-1]
            loss_total = loss_total + args.bce_loss_weight * final_mask_loss
            loss_total = loss_total + args.lambda_intermediate * intermediate_loss
            intermediate_meter.update(intermediate_loss.item())

        diag = compute_refinement_diagnostics(out)
        refine_delta_meter.update(diag["refine_delta_logit_mean"])

        loss_total.backward()
        optimizer.step()
        loss_meter.update(loss_total.item())

    return {
        "loss": loss_meter.avg,
        "intermediate_mask_loss": intermediate_meter.avg,
        "refine_delta_logit_mean": refine_delta_meter.avg,
    }


def evaluate(test_loader, model, args, save_result=False, best_epoch=None):
    global device
    model.eval()
    loss_meter = AverageMeter()
    cd_after_meter = AverageMeter()
    avg_shift_last_meter = AverageMeter()
    total_disp_meter = AverageMeter()
    disp_std_meter = AverageMeter()
    intermediate_meter = AverageMeter()
    refine_delta_meter = AverageMeter()
    prob_shift_meter = AverageMeter()
    gate_entropy_meter = AverageMeter()

    all_probs_final = []
    all_probs_step0 = []
    all_labels = []

    outdir = args.checkpoint.split('/')[-1]
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = forward_model(model, data)
            y_pred = out["x_offset"] + data.pos
            logits_steps = out["logits_steps"]
            mask_prob = out["mask_prob"]
            bandwidth = out["bandwidth"]

            loss_total = 0.0
            num_graphs = len(torch.unique(data.batch))

            for i in range(num_graphs):
                joint_gt = data.joints[data.joints_batch == i, :]
                y_pred_i = y_pred[data.batch == i, :]
                mask_prob_i = mask_prob[data.batch == i]

                cd_before, cd_after, loss_ms, clustered_pred = joint_meanshift_loss(
                    y_pred_i, mask_prob_i, bandwidth, joint_gt, args
                )
                loss_total += cd_before + args.ms_loss_weight * loss_ms
                cd_after_meter.update(cd_after.item())

                if len(clustered_pred) >= 2:
                    shift_last = torch.norm(clustered_pred[-1] - clustered_pred[-2], dim=1).mean().item()
                else:
                    shift_last = 0.0
                avg_shift_last_meter.update(shift_last)

                y_pred_final = clustered_pred[-1]
                total_disp = torch.norm(y_pred_final - y_pred_i, dim=1).mean().item()
                total_disp_meter.update(total_disp)
                disp_std_meter.update(torch.norm(y_pred_final - y_pred_i, dim=1).std().item())

                if save_result:
                    output_folder = 'results/{:s}/best_{:d}/'.format(outdir, best_epoch)
                    if not os.path.exists(output_folder):
                        mkdir_p(output_folder)
                    output_point_cloud_ply(y_pred_i, name=str(data.name[i].item()), output_folder=output_folder)
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

                all_probs_final.append(torch.sigmoid(logits_steps[-1]).detach().cpu().numpy().reshape(-1))
                all_probs_step0.append(torch.sigmoid(logits_steps[0]).detach().cpu().numpy().reshape(-1))
                all_labels.append(mask_gt.detach().cpu().numpy().reshape(-1))

            diag = compute_refinement_diagnostics(out)
            refine_delta_meter.update(diag["refine_delta_logit_mean"])
            prob_shift_meter.update(diag["mask_prob_shift_mean"])
            gate_entropy_meter.update(diag["gate_entropy_mean"])

            loss_meter.update(loss_total.item())

    metrics = {
        'cd_after': cd_after_meter.avg,
        'avg_shift_last': avg_shift_last_meter.avg,
        'total_disp': total_disp_meter.avg,
        'disp_std': disp_std_meter.avg,
        'intermediate_mask_loss': intermediate_meter.avg,
        'refine_delta_logit_mean': refine_delta_meter.avg,
        'mask_prob_shift_mean': prob_shift_meter.avg,
        'gate_entropy_mean': gate_entropy_meter.avg,
    }

    if args.use_bce and all_labels:
        final_cls = compute_mask_classification_metrics(
            np.concatenate(all_probs_final), np.concatenate(all_labels)
        )
        step0_cls = compute_mask_classification_metrics(
            np.concatenate(all_probs_step0), np.concatenate(all_labels)
        )
        metrics.update(final_cls)
        metrics["mask_f1_step0"] = step0_cls["mask_f1"]
        metrics["mask_f1_improvement"] = final_cls["mask_f1"] - step0_cls["mask_f1"]
        metrics["mask_pr_auc_step0"] = step0_cls["mask_pr_auc"]

    return loss_meter.avg, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint finetune with iterative mask refinement')
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
    parser.add_argument('--jointnet_lr', default=5e-5, type=float)
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
