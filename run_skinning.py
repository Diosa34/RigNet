#-------------------------------------------------------------------------------
# Name:        run_skinning.py
# Purpose:     Train a network to predict skinning weights
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import os
import sys
sys.path.append("./")
import shutil
import argparse
import numpy as np

from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import output_rigging
from utils.log_args_to_mlflow import log_args_to_mlflow

import torch
import torch.backends.cudnn as cudnn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from models.supplemental_layers.cross_entropy_with_probs import cross_entropy_with_probs
from datasets.skin_dataset import SkinDataset

import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def get_bone_names(data_folder, sample_name):
    """Load bone names from skin.txt file located in data_folder."""
    skin_filename = os.path.join(data_folder, "{:d}_skin.txt".format(sample_name))
    with open(skin_filename, 'r') as fin:
        lines = fin.readlines()
    bone_names = []
    for li in lines:
        words = li.strip().split()
        if words[0] == 'bones':
            bone_names.append([words[1], words[2]])
    return bone_names


def post_filter(skin_weights, topology_edge, num_ring=1):
    skin_weights_new = np.zeros_like(skin_weights)
    for v in range(len(skin_weights)):
        adj_verts_multi_ring = []
        current_seeds = [v]
        for r in range(num_ring):
            adj_verts = []
            for seed in current_seeds:
                adj_edges = topology_edge[:, np.argwhere(topology_edge == seed)[:, 1]]
                adj_verts_seed = list(set(adj_edges.flatten().tolist()))
                adj_verts_seed.remove(seed)
                adj_verts += adj_verts_seed
            adj_verts_multi_ring += adj_verts
            current_seeds = adj_verts
        adj_verts_multi_ring = list(set(adj_verts_multi_ring))
        if v in adj_verts_multi_ring:
            adj_verts_multi_ring.remove(v)
        skin_weights_neighbor = [skin_weights[int(i), :][np.newaxis, :] for i in adj_verts_multi_ring]
        skin_weights_neighbor = np.concatenate(skin_weights_neighbor, axis=0)
        #max_bone_id = np.argmax(skin_weights[v, :])
        #if np.sum(skin_weights_neighbor[:, max_bone_id]) < 0.17 * len(skin_weights_neighbor):
        #    skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)
        #else:
        #    skin_weights_new[v, :] = skin_weights[v, :]
        skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)

    #skin_weights_new[skin_weights_new.sum(axis=1) == 0, :] = skin_weights[skin_weights_new.sum(axis=1) == 0, :]
    return skin_weights_new


def grid_search_threshold(val_loader, model, args, thresholds=None):
    """
    Perform grid search over thresholds to find best threshold based on F1 score on validation set.
    Returns best_threshold, best_metrics (dict with loss, L1, avg_dist, max_dist, precision, recall, f1)
    """
    if thresholds is None:
        thresholds = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3]
    best_f1 = -1
    best_th = None
    best_metrics = None
    results = []
    for th in thresholds:
        # temporarily override threshold
        orig_th = args.threshold
        args.threshold = th
        metrics = test(val_loader, model, args, save_result=False)
        f1 = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-10)
        metrics['f1'] = f1
        results.append((th, metrics, f1))
        print(f"Threshold {th:.2e}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = metrics
        args.threshold = orig_th  # restore
    return best_th, best_metrics, results


def main(args):
    global device
    lowest_loss = 1e20

    # create checkpoint dir and log dir
    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume:
        if isdir(args.logdir):
            shutil.rmtree(args.logdir)
        mkdir_p(args.logdir)

    # create model
    model = models.__dict__["skinnet"](args.nearest_bone, args.Dg, args.Lf)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    lr = args.lr
    best_threshold = args.threshold  # default from args
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            # load best_threshold if saved
            if 'best_threshold' in checkpoint:
                best_threshold = checkpoint['best_threshold']
                args.threshold = best_threshold
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(SkinDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=True)
    val_loader = DataLoader(SkinDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False)
    test_loader = DataLoader(SkinDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False)

    if args.evaluate:
        print('\nEvaluation only')
        # If grid_search is requested, find best threshold on val_loader and use it for test
        if args.grid_search:
            print("Performing grid search on validation set to find best threshold...")
            best_th, best_val_metrics, _ = grid_search_threshold(val_loader, model, args)
            print(f"Best threshold: {best_th:.2e}")
            print(f"Best validation metrics: {best_val_metrics}")
            # update args.threshold to best found
            args.threshold = best_th
            # Optionally log to MLflow if run active (we are in evaluate, no active run unless we start one)
            # We can start a run for evaluation
            mlflow.set_experiment(f"RigNet_skinning")
            with mlflow.start_run(run_name="evaluation_gridsearch", nested=True):
                mlflow.log_param("best_threshold", best_th)
                for k, v in best_val_metrics.items():
                    mlflow.log_metric(f"val_best_{k}", v)
                # now test with best threshold
                test_metrics = test(test_loader, model, args, save_result=True)
                print("Test metrics with best threshold:")
                for k, v in test_metrics.items():
                    print(f"test_{k}: {v:.6f}")
                    mlflow.log_metric(f"test_{k}", v)
        else:
            test_metrics = test(test_loader, model, args, save_result=True)
            print("test_loss: {:.6f}".format(test_metrics['loss']))
            for k, v in test_metrics.items():
                if k != 'loss':
                    print("test_{}: {:.6f}".format(k, v))
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)

    mlflow.set_experiment(f"RigNet_skinning")
    with mlflow.start_run(run_name=f"skinning"):
        log_args_to_mlflow(args)
        mlflow.log_param("device", str(device))
        for epoch in range(args.start_epoch, args.epochs):
            lr = scheduler.get_last_lr()
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr[0]))
            train_loss = train(train_loader, model, optimizer, args)
            val_metrics = test(val_loader, model, args)
            test_metrics = test(test_loader, model, args)
            val_loss = val_metrics["loss"]
            test_loss = test_metrics["loss"]
            scheduler.step()
            print('Epoch{:d}. train_loss: {:.6f}.'.format(epoch + 1, train_loss))
            print('Epoch{:d}. val_loss: {:.6f}.'.format(epoch + 1, val_loss))
            print('Epoch{:d}. test_loss: {:.6f}.'.format(epoch + 1, test_loss))

            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v), step=epoch + 1)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", float(v), step=epoch + 1)
            mlflow.log_metric("lr", lr[0], step=epoch + 1)

            # remember best acc and save checkpoint
            is_best = val_loss < lowest_loss
            lowest_loss = min(val_loss, lowest_loss)
            # Save checkpoint with current threshold (which may be updated later)
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss,
                             'optimizer': optimizer.state_dict(),
                             'best_threshold': args.threshold},  # store current threshold
                            is_best, checkpoint=args.checkpoint)

            # TensorBoard logging
            logger.add_scalar("train/loss", train_loss, epoch + 1)
            for k, v in val_metrics.items():
                logger.add_scalar(f"val/{k}", float(v), epoch + 1)
            for k, v in test_metrics.items():
                logger.add_scalar(f"test/{k}", float(v), epoch + 1)

        # After training, load best model and perform grid search on validation set to find optimal threshold
        best_model_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
        if isfile(best_model_path):
            print("=> loading best model '{}' for threshold tuning".format(best_model_path))
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
            # perform grid search on validation set
            print("Performing grid search on validation set to find best threshold...")
            best_th, best_val_metrics, _ = grid_search_threshold(val_loader, model, args)
            print(f"Best threshold: {best_th:.2e}")
            print(f"Best validation metrics: {best_val_metrics}")
            # update args.threshold and also save it in checkpoint for future use
            args.threshold = best_th
            # Save updated checkpoint with best_threshold
            checkpoint['best_threshold'] = best_th
            torch.save(checkpoint, os.path.join(args.checkpoint, 'model_best.pth.tar'))
            print("Testing with best threshold and saving results...")
            test_metrics = test(test_loader, model, args, save_result=True)
            print("Test metrics with best threshold:")
            for k, v in test_metrics.items():
                print(f"test_{k}: {v:.6f}")
            mlflow.log_param("best_threshold", best_th)
        else:
            print("No best model found, skipping threshold tuning.")


def train(train_loader, model, optimizer, args):
    global device
    model.train()  # switch to train mode
    loss_meter = AverageMeter()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        skin_pred = model(data)
        skin_gt = data.skin_label[:, 0:args.nearest_bone]
        loss_mask_batch = data.loss_mask.float()[:, 0:args.nearest_bone]
        skin_gt = skin_gt * loss_mask_batch
        skin_gt = skin_gt / (torch.sum(torch.abs(skin_gt), dim=1, keepdim=True) + 1e-8)
        vert_mask = (torch.abs(skin_gt.sum(dim=1) - 1.0) < 1e-8).float()  # mask out vertices whose skinning is missing from the picked K bones.
        loss = cross_entropy_with_probs(skin_pred, skin_gt, reduction='none')
        loss = (loss * loss_mask_batch * vert_mask.unsqueeze(1)).sum() / (loss_mask_batch * vert_mask.unsqueeze(1)).sum()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    return loss_meter.avg


def test(test_loader, model, args, save_result=False):
    """
    Evaluate model on a given dataloader.
    Computes loss and additional metrics (L1, avg_dist, max_dist, precision, recall).
    Files (*_skin.txt, *_tpl_e.txt) are loaded from test_loader.dataset.root.
    """
    global device
    model.eval()
    loss_meter = AverageMeter()

    # accumulators for metrics
    all_l1 = []
    all_avg_dist = []
    all_max_dist = []
    all_precision = []
    all_recall = []

    threshold = args.threshold
    outdir = args.checkpoint.split('/')[-1]

    # Data folder (root of the dataset)
    data_folder = test_loader.dataset.root

    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            skin_pred = model(data)  # logits, shape (total_V, K)
            skin_gt = data.skin_label[:, 0:args.nearest_bone]
            loss_mask_batch = data.loss_mask.float()[:, 0:args.nearest_bone]
            skin_gt = skin_gt * loss_mask_batch
            skin_gt = skin_gt / (torch.sum(torch.abs(skin_gt), dim=1, keepdim=True) + 1e-8)
            vert_mask = (torch.abs(skin_gt.sum(dim=1) - 1.0) < 1e-8).float()
            loss = cross_entropy_with_probs(skin_pred, skin_gt, reduction='none')
            loss = (loss * loss_mask_batch * vert_mask.unsqueeze(1)).sum() / (loss_mask_batch * vert_mask.unsqueeze(1)).sum()
            loss_meter.update(loss.item())

            # Process each sample in the batch
            batch_indices = torch.unique(data.batch)
            for i in batch_indices:
                mask = (data.batch == i)
                sample_name = data.name[i].item()

                # predicted logits and nearest indices for this sample
                pred_logits_i = skin_pred[mask]                  # (V_i, K)
                skin_nn_i = data.skin_nn[mask]                   # (V_i, K)
                loss_mask_i = data.loss_mask[mask, :args.nearest_bone]  # (V_i, K)

                # softmax and apply mask, then renormalize
                pred_probs_i = torch.softmax(pred_logits_i, dim=1)      # (V_i, K)
                pred_probs_i = pred_probs_i * loss_mask_i
                pred_probs_i = pred_probs_i / (pred_probs_i.sum(dim=1, keepdim=True) + 1e-10)

                # load bone names to get actual number of bones
                bone_names = get_bone_names(data_folder, sample_name)
                num_bones = len(bone_names)

                # ground truth weights for K nearest bones (already normalized)
                skin_gt_i = skin_gt[mask]  # (V_i, K)

                # construct full prediction and ground truth vectors
                V_i = pred_probs_i.size(0)
                pred_full_i = torch.zeros((V_i, num_bones), device=device)
                gt_full_i = torch.zeros((V_i, num_bones), device=device)
                for v in range(V_i):
                    for nn_id in range(args.nearest_bone):
                        bone_idx = skin_nn_i[v, nn_id].item()
                        if bone_idx < num_bones:
                            pred_full_i[v, bone_idx] = pred_probs_i[v, nn_id]
                            gt_full_i[v, bone_idx] = skin_gt_i[v, nn_id]
                # renormalize
                pred_full_i = pred_full_i / (pred_full_i.sum(dim=1, keepdim=True) + 1e-10)
                gt_full_i = gt_full_i / (gt_full_i.sum(dim=1, keepdim=True) + 1e-10)

                # convert to numpy for metric computation
                pred_full_np = pred_full_i.cpu().numpy()
                gt_full_np = gt_full_i.cpu().numpy()

                # compute metrics (as in RigNet paper)
                # L1 error per vertex (sum over bones), then mean over vertices
                l1_per_vertex = np.sum(np.abs(pred_full_np - gt_full_np), axis=1)
                l1_mean = np.mean(l1_per_vertex)
                all_l1.append(l1_mean)

                # L2 distance per vertex
                l2_per_vertex = np.sqrt(np.sum((pred_full_np - gt_full_np) ** 2, axis=1))
                avg_dist = np.mean(l2_per_vertex)
                max_dist = np.max(l2_per_vertex)
                all_avg_dist.append(avg_dist)
                all_max_dist.append(max_dist)

                # Precision and Recall with threshold (active bones)
                pred_active = (pred_full_np > threshold)
                gt_active = (gt_full_np > threshold)
                TP = np.sum(pred_active & gt_active)
                FP = np.sum(pred_active & ~gt_active)
                FN = np.sum(~pred_active & gt_active)
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                all_precision.append(precision)
                all_recall.append(recall)

                if save_result:
                    output_folder = 'results/{:s}/'.format(outdir)
                    if not os.path.exists(output_folder):
                        mkdir_p(output_folder)

                    # load topology edges for post-filter
                    tpl_e_path = os.path.join(data_folder, "{:d}_tpl_e.txt".format(sample_name))
                    tpl_e = np.loadtxt(tpl_e_path).T

                    # post-filter and threshold
                    skin_pred_asarray = post_filter(pred_full_np, tpl_e, num_ring=1)
                    skin_pred_asarray[skin_pred_asarray < np.max(skin_pred_asarray, axis=1, keepdims=True) * 0.5] = 0.0
                    skin_pred_asarray = skin_pred_asarray / (skin_pred_asarray.sum(axis=1, keepdims=True) + 1e-10)

                    # save bone names
                    with open(os.path.join(output_folder, "{:d}_bone_names.txt".format(sample_name)), 'w') as fout:
                        for bone_name in bone_names:
                            fout.write("{:s} {:s}\n".format(bone_name[0], bone_name[1]))
                    np.save(os.path.join(output_folder, "{:d}_full_pred.npy".format(sample_name)), skin_pred_asarray)

                    # output rigging (generates .skel and .weight files)
                    skel_filename = os.path.join(args.info_folder, "{:d}.txt".format(sample_name))
                    output_rigging(skel_filename, skin_pred_asarray, output_folder, sample_name)

    # compute average metrics over all samples
    metrics = {
        "loss": loss_meter.avg,
        "L1": np.mean(all_l1) if all_l1 else 0.0,
        "avg_dist": np.mean(all_avg_dist) if all_avg_dist else 0.0,
        "max_dist": np.mean(all_max_dist) if all_max_dist else 0.0,
        "precision": np.mean(all_precision) if all_precision else 0.0,
        "recall": np.mean(all_recall) if all_recall else 0.0,
    }
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='skinning predition network')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[], help='Decrease learning rate at these epochs.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on val/test set')
    ####################################################################################################################
    parser.add_argument('--train_batch', default=2, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--test_batch', default=2, type=int, metavar='N', help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/test', type=str, metavar='LOG', help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/train/',
                        type=str, help='folder of training data')
    parser.add_argument('--val_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/val/',
                        type=str, help='folder of validation data')
    parser.add_argument('--test_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/test/',
                        type=str, help='folder of testing data')
    parser.add_argument('--nearest_bone', type=int, default=5)
    parser.add_argument('--info_folder', default='/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/rig_info_remesh/',
                        type=str, help='folder of skeleton information')
    parser.add_argument('--Dg', action='store_true', help='input inverset geodesic as addtional feature')
    parser.add_argument('--Lf', action='store_true', help='input isleaf indicator as addtional feature')
    # Threshold for metrics
    parser.add_argument('--threshold', default=1e-4, type=float, help='threshold for active bones in precision/recall')
    # Grid search flag
    parser.add_argument('--grid_search', action='store_true', help='run grid search over thresholds to optimize F1 on validation set')
    print(parser.parse_args())
    main(parser.parse_args())