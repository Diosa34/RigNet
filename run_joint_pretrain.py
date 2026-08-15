#-------------------------------------------------------------------------------
# Name:        run_joint_pretrain.py
# Purpose:     Pretrain regression module and attation module for joint prediction
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
sys.path.append("./")
import os
import shutil
import argparse
import numpy as np

from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import output_point_cloud_ply
from utils.log_args_to_mlflow import log_args_to_mlflow

import torch
import torch.backends.cudnn as cudnn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.skeleton_dataset import GraphDataset
from models.GCN import JointPredNet
from models.moe_modules import collect_moe_load_balance_loss, collect_moe_train_metrics
from models.supplemental_layers.pytorch_chamfer_dist import chamfer_distance_with_average

import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from scipy.spatial.distance import cdist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


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
    if args.arch == 'jointnet':
        model = JointPredNet(
            out_channels=3, input_normal=args.input_normal, arch=args.arch, aggr=args.aggr,
            use_moe=args.use_moe, num_experts=args.num_experts, top_k=args.top_k,
        )
    elif args.arch == 'masknet':
        model = JointPredNet(
            out_channels=1, input_normal=args.input_normal, arch=args.arch, aggr=args.aggr,
            use_moe=args.use_moe, num_experts=args.num_experts, top_k=args.top_k,
        )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(GraphDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=True, follow_batch=['joints'])
    val_loader = DataLoader(GraphDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
    test_loader = DataLoader(GraphDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
    if args.evaluate:
        print('\nEvaluation only')
        test_metrics = test( test_loader, model, args, save_result=True, best_epoch=args.start_epoch ) 
        print(test_metrics)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)

    mlflow.set_experiment(f"RigNet_{args.arch}_pretrain")
    with mlflow.start_run(run_name=f"joint_{args.arch}_pretrain"):
        log_args_to_mlflow(args)
        mlflow.log_param("device", str(device))
        for epoch in range(args.start_epoch, args.epochs):
            lr = scheduler.get_last_lr()
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr[0]))
            train_loss, moe_metrics = train(train_loader, model, optimizer, args)
            val_metrics  = test(val_loader, model, args)
            test_metrics = test(test_loader, model, args)
            val_loss = val_metrics["loss"]
            test_loss = test_metrics["loss"]
            
            scheduler.step()
            print('Epoch{:d}. train_loss: {:.6f}.'.format(epoch + 1, train_loss))
            print('Epoch{:d}. val_loss: {:.6f}.'.format(epoch + 1, val_loss))
            print('Epoch{:d}. test_loss: {:.6f}.'.format(epoch + 1, test_loss))

            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            if moe_metrics:
                mlflow.log_metric(
                    "train_moe_load_balance_loss",
                    moe_metrics["load_balance_loss"],
                    step=epoch + 1,
                )
                mlflow.log_metric(
                    "train_moe_expert_usage_spread",
                    moe_metrics["expert_usage_spread"],
                    step=epoch + 1,
                )
            for k, v in val_metrics.items():
                mlflow.log_metric(
                    f"val_{k}",
                    float(v),
                    step=epoch + 1
                )
            for k, v in test_metrics.items():
                mlflow.log_metric(
                    f"test_{k}",
                    float(v),
                    step=epoch + 1
                )
            mlflow.log_metric("lr", lr[0], step=epoch + 1)
            
            # remember best acc and save checkpoint
            is_best = val_loss < lowest_loss
            lowest_loss = min(val_loss, lowest_loss)
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss, 'optimizer': optimizer.state_dict()},
                            is_best, checkpoint=args.checkpoint)
    
            logger.add_scalar(
                "train/loss",
                train_loss,
                epoch + 1
            )
            if moe_metrics:
                logger.add_scalar(
                    "train/moe_load_balance_loss",
                    moe_metrics["load_balance_loss"],
                    epoch + 1,
                )
                logger.add_scalar(
                    "train/moe_expert_usage_spread",
                    moe_metrics["expert_usage_spread"],
                    epoch + 1,
                )
            
            for k, v in val_metrics.items():
                logger.add_scalar(
                    f"val/{k}",
                    float(v),
                    epoch + 1
                )
            
            for k, v in test_metrics.items():
                logger.add_scalar(
                    f"test/{k}",
                    float(v),
                    epoch + 1
                )

        best_model_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
        print("=> loading checkpoint '{}'".format(best_model_path))
        checkpoint = torch.load(best_model_path)
        best_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(best_model_path, best_epoch))
        test_metrics = test(
            test_loader,
            model,
            args,
            save_result=True,
            best_epoch=best_epoch
        )
        
        print(f"Best epoch: {best_epoch}")
        print(f"test_loss: {test_metrics['loss']:.6f}")
        
        for k, v in test_metrics.items():
            print(f"{k}: {v}")


def train(train_loader, model, optimizer, args):
    global device
    model.train()  # switch to train mode
    loss_meter = AverageMeter()
    lb_loss_meter = AverageMeter()
    usage_spread_meter = AverageMeter()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if args.arch == 'masknet':
            mask_pred = model(data)
            mask_gt = data.mask.unsqueeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(mask_pred, mask_gt.float(), reduction='mean')
        elif args.arch == 'jointnet':
            data_displacement = model(data)
            y_pred = data_displacement + data.pos
            loss = 0.0
            for i in range(len(torch.unique(data.joints_batch))):
                joint_gt = data.joints[data.joints_batch == i, :]
                y_pred_i = y_pred[data.batch == i, :]
                loss += chamfer_distance_with_average(y_pred_i.unsqueeze(0), joint_gt.unsqueeze(0))
            num_graphs = len(torch.unique(data.batch))
            loss /= num_graphs
        if args.use_moe:
            moe_step_metrics = collect_moe_train_metrics(model)
            if 'load_balance_loss' in moe_step_metrics:
                lb_loss_meter.update(moe_step_metrics['load_balance_loss'])
            if 'expert_usage_spread' in moe_step_metrics:
                usage_spread_meter.update(moe_step_metrics['expert_usage_spread'])
            if args.moe_lb_weight > 0 and 'load_balance_loss' in moe_step_metrics:
                lb_loss = collect_moe_load_balance_loss(model)
                if not torch.is_tensor(lb_loss):
                    lb_loss = torch.tensor(lb_loss, device=device)
                loss = loss + args.moe_lb_weight * lb_loss
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    moe_metrics = {}
    if args.use_moe and lb_loss_meter.count > 0:
        moe_metrics['load_balance_loss'] = lb_loss_meter.avg
        print('  moe_load_balance_loss: {:.6f} (target ~{:.1f} for balanced routing)'.format(
            lb_loss_meter.avg, 3.0))
    if args.use_moe and usage_spread_meter.count > 0:
        moe_metrics['expert_usage_spread'] = usage_spread_meter.avg
        print('  moe_expert_usage_spread: {:.4f} (higher => more specialized routing)'.format(
            usage_spread_meter.avg))
    return loss_meter.avg, moe_metrics


def test(test_loader, model, args, save_result=False, best_epoch=None):
    global device
    model.eval()  # switch to test mode
    loss_meter = AverageMeter()
    
    metrics = {}

    # masknet
    all_probs = []
    all_labels = []
    
    # jointnet
    joint_cd = []
    joint_mean_error = []
    
    joint_recall_001 = []
    joint_recall_0025 = []
    joint_recall_005 = []
    
    outdir = args.checkpoint.split('/')[-1]
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            if args.arch == 'masknet':
                mask_pred = model(data)
                mask_gt = data.mask.unsqueeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(mask_pred, mask_gt.float(), reduction='mean')
                prob = torch.sigmoid(mask_pred)

                all_probs.append(
                    prob.detach().cpu().numpy().reshape(-1)
                )
                
                all_labels.append(
                    mask_gt.detach().cpu().numpy().reshape(-1)
                )
            elif args.arch == 'jointnet':
                data_displacement = model(data)
                y_pred = data_displacement + data.pos
                loss = 0.0

                for i in range(len(torch.unique(data.joints_batch))):
                
                    joint_gt = data.joints[data.joints_batch == i, :]
                    y_pred_i = y_pred[data.batch == i, :]
                
                    cd = chamfer_distance_with_average(
                        y_pred_i.unsqueeze(0),
                        joint_gt.unsqueeze(0)
                    )
                
                    loss += cd
                
                    gt = joint_gt.detach().cpu().numpy()
                    pred = y_pred_i.detach().cpu().numpy()
                
                    dist_matrix = cdist(gt, pred)
                
                    min_gt_to_pred = dist_matrix.min(axis=1)
                
                    joint_cd.append(cd.item())
                
                    joint_mean_error.append(
                        min_gt_to_pred.mean()
                    )
                
                    joint_recall_001.append(
                        (min_gt_to_pred < 0.01).mean()
                    )
                
                    joint_recall_0025.append(
                        (min_gt_to_pred < 0.025).mean()
                    )
                
                    joint_recall_005.append(
                        (min_gt_to_pred < 0.05).mean()
                    )
                
                num_graphs = len(torch.unique(data.batch))
                loss /= num_graphs
            loss_meter.update(loss.item())

            if save_result:
                output_folder = 'results/{:s}/best_{:d}/'.format(outdir, best_epoch)
                if not os.path.exists(output_folder):
                    mkdir_p(output_folder)
                if args.arch == 'masknet':
                    mask_pred = torch.sigmoid(mask_pred)
                    for i in range(len(torch.unique(data.batch))):
                        mask_pred_sample = mask_pred[data.batch == i]
                        np.save(os.path.join(output_folder, str(data.name[i].item()) + '_attn.npy'), mask_pred_sample.data.to("cpu").numpy())
                else:
                    for i in range(len(torch.unique(data.batch))):
                        y_pred_sample = y_pred[data.batch == i, :]
                        output_point_cloud_ply(y_pred_sample, name=str(data.name[i].item()),
                                               output_folder='results/{:s}/best_{:d}/'.format(outdir, best_epoch))
    metrics["loss"] = loss_meter.avg

    if args.arch == "jointnet":

        metrics["chamfer"] = np.mean(joint_cd)
    
        metrics["mean_joint_error"] = np.mean(
            joint_mean_error
        )
    
        metrics["recall_0.01"] = np.mean(
            joint_recall_001
        )
    
        metrics["recall_0.025"] = np.mean(
            joint_recall_0025
        )
    
        metrics["recall_0.05"] = np.mean(
            joint_recall_005
        )
    if args.arch == "masknet":
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
    
        pred_bin = (all_probs > 0.5).astype(np.uint8)
    
        metrics["precision"] = precision_score(
            all_labels,
            pred_bin,
            zero_division=0
        )
    
        metrics["recall"] = recall_score(
            all_labels,
            pred_bin,
            zero_division=0
        )
    
        metrics["f1"] = f1_score(
            all_labels,
            pred_bin,
            zero_division=0
        )
    
        metrics["pr_auc"] = average_precision_score(
            all_labels,
            all_probs
        )
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyG DGCNN')
    parser.add_argument('--arch', default='masknet')  # jointnet, masknet
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[], help='Decrease learning rate at these epochs.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on val/test set')
    parser.add_argument('--input_normal', action='store_true')
    parser.add_argument('--aggr', default='max', type=str)
    parser.add_argument('--no-use-moe', dest='use_moe', action='store_false', default=True, help='Disable MoE and use original MLP blocks')
    parser.add_argument('--num-experts', default=4, type=int, help='number of MoE experts')
    parser.add_argument('--top-k', default=2, type=int, help='top-k experts per token')
    parser.add_argument('--moe-lb-weight', default=0.01, type=float,
                        help='auxiliary load-balancing loss weight (Switch Transformer style)')
    ######################
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
    print(parser.parse_args())
    main(parser.parse_args())
