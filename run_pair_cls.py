#-------------------------------------------------------------------------------
# Name:        run_pair_cls.py
# Purpose:     Train a network (bonenet) to predict pair-wise connectivity cost
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

import torch
import torch.backends.cudnn as cudnn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.PairCls_GCN import PairCls
from models.moe_modules import collect_moe_load_balance_loss, collect_moe_train_metrics
from datasets.skeleton_dataset import GraphDataset
from utils.os_utils import isdir, mkdir_p, isfile
from utils.log_utils import AverageMeter
from utils.log_args_to_mlflow import log_args_to_mlflow

import mlflow
import mlflow.pytorch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

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
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
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
    
    model = PairCls(use_moe=args.use_moe, num_experts=args.num_experts, top_k=args.top_k)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(GraphDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=True, follow_batch=['joints', 'pairs'])
    val_loader = DataLoader(GraphDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints', 'pairs'])
    test_loader = DataLoader(GraphDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints', 'pairs'])

    if args.evaluate:
        print('\nEvaluation only')
        test_metrics = test(test_loader, model, args, save_result=True, best_epoch=args.start_epoch)
        print(test_metrics)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)

    mlflow.set_experiment(f"RigNet_bonenet")
    with mlflow.start_run(run_name=f"joint_bonenet{'_moe' if args.use_moe else ''}"):
        log_args_to_mlflow(args)
        mlflow.log_param("device", str(device))
        for epoch in range(args.start_epoch, args.epochs):
            lr = scheduler.get_last_lr()
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr[0]))

            train_loss, moe_metrics = train(train_loader, model, optimizer, args)
            
            val_metrics = test(val_loader, model, args)
            val_loss = val_metrics["loss"]
            
            test_metrics = test(test_loader, model, args, best_epoch=epoch+1)
            test_loss = test_metrics["loss"]
            
            scheduler.step()
            
            print('Epoch{:d}. train_loss: {:.6f}.'.format(epoch + 1, train_loss))
            print('Epoch{:d}. val_loss: {:.6f}.'.format(epoch + 1, val_loss))
            print('Epoch{:d}. test_loss: {:.6f}.'.format(epoch + 1, test_loss))
            
            # log metrics to MLflow
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
                mlflow.log_metric(f"val_{k}", float(v), step=epoch + 1)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", float(v), step=epoch + 1)
            mlflow.log_metric("lr", lr[0], step=epoch + 1)
            
            # log metrics to TensorBoard
            logger.add_scalar("train/loss", train_loss, epoch + 1)
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
                logger.add_scalar(f"val/{k}", float(v), epoch + 1)
            for k, v in test_metrics.items():
                logger.add_scalar(f"test/{k}", float(v), epoch + 1)

            is_best = val_loss < lowest_loss
            lowest_loss = min(val_loss, lowest_loss)
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss, 'optimizer': optimizer.state_dict()},
                            is_best, checkpoint=args.checkpoint)

        best_model_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
        print("=> loading checkpoint '{}'".format(best_model_path))
        checkpoint = torch.load(best_model_path)
        best_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(best_model_path, best_epoch))
        test_metrics = test(test_loader, model, args, save_result=True, best_epoch=best_epoch)

        print(f"Best epoch: {best_epoch}")
        print(f"test_loss: {test_metrics['loss']:.6f}")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

        mlflow.log_metric("best_epoch", best_epoch)


def train(train_loader, model, optimizer, args):
    global device
    model.train()  # switch to train mode
    loss_meter = AverageMeter()
    lb_loss_meter = AverageMeter()
    usage_spread_meter = AverageMeter()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pre_label, label = model(data)

        loss1 = torch.nn.functional.binary_cross_entropy_with_logits(pre_label, label, reduction='none')
        topk_val, _ = torch.topk(loss1.view(-1), k=int(args.topk * len(pre_label)), dim=0, sorted=False)
        loss2 = topk_val.mean()
        loss = loss1.mean() + loss2
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
    if args.use_moe and usage_spread_meter.count > 0:
        moe_metrics['expert_usage_spread'] = usage_spread_meter.avg
    return loss_meter.avg, moe_metrics


def test(test_loader, model, args, save_result=False, best_epoch=None):
    global device
    model.eval()  # switch to test mode
    if save_result:
        output_folder = 'results/moe/{:s}/best_{:d}/'.format(args.checkpoint.split('/')[-1], best_epoch)
        if not os.path.exists(output_folder):
            mkdir_p(output_folder)

    loss_meter = AverageMeter()
    all_probs = []
    all_labels = []

    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pre_label, label = model(data)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pre_label, label.float())
            loss_meter.update(loss.item())

            prob = torch.sigmoid(pre_label).view(-1)
            all_probs.append(prob.detach().cpu().numpy())
            all_labels.append(label.view(-1).detach().cpu().numpy())

            if save_result:
                connect_prob = torch.sigmoid(pre_label)
                acc_joints = 0
                for i in range(len(torch.unique(data.batch))):
                    pair_idx = data.pairs[data.pairs_batch==i].long()
                    connect_prob_i = connect_prob[data.pairs_batch==i]
                    num_joint = len(data.joints[data.joints_batch==i])
                    cost_matrix = np.zeros((num_joint, num_joint))
                    pair_idx = pair_idx.to("cpu").numpy()
                    cost_matrix[pair_idx[:, 0]-acc_joints, pair_idx[:, 1]-acc_joints] = connect_prob_i.data.cpu().numpy().squeeze(axis=1)
                    cost_matrix = 1 - cost_matrix
                    print('saving: {:s}'.format(str(data.name[i].item()) + '_cost.npy'))
                    np.save(os.path.join(output_folder, str(data.name[i].item()) + '_cost.npy'), cost_matrix)
                    acc_joints += num_joint

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    pred_bin = (all_probs > 0.5).astype(np.uint8)

    metrics = {
        "loss": loss_meter.avg,
        "accuracy": accuracy_score(all_labels, pred_bin),
        "precision": precision_score(all_labels, pred_bin, zero_division=0),
        "recall": recall_score(all_labels, pred_bin, zero_division=0),
        "f1": f1_score(all_labels, pred_bin, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs)
    }

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='joint connectivity')
    parser.add_argument('--arch', default='paircls')  # paircls_fc, paircls_nogt, paircls_nogs
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200], help='Decrease learning rate at these epochs.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    ####################################################################################################################
    parser.add_argument('--train_batch', default=2, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--test_batch', default=2, type=int, metavar='N', help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/moe/connect_test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/moe/connect_test', type=str, metavar='LOG', help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_folder', default='/home/jovyan/data-storage/ModelResource_RigNetv1_preproccessed/train/',
                        type=str, help='folder of training data')
    parser.add_argument('--val_folder', default='/home/jovyan/data-storage/ModelResource_RigNetv1_preproccessed/val/',
                        type=str, help='folder of validation data')
    parser.add_argument('--test_folder', default='/home/jovyan/data-storage/ModelResource_RigNetv1_preproccessed/test/',
                        type=str, help='folder of testing data')
    
    parser.add_argument('--topk', default=0.3, type=float, help='topk ratio for ohem')
    parser.add_argument('--no-use-moe', dest='use_moe', action='store_false', default=True)
    parser.add_argument('--num-experts', default=4, type=int)
    parser.add_argument('--top-k', default=2, type=int)
    parser.add_argument('--moe-lb-weight', default=0.01, type=float)
    parser.add_argument('--cuda', default=0, type=int, help='CUDA device index')
    args = parser.parse_args()
    print(args)
    main(args)
