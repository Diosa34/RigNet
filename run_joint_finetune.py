#-------------------------------------------------------------------------------
# Name:        run_joint_finetune.py
# Purpose:     Finetuning regression and attention modules together with a meanshift module
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

import torch
import torch.backends.cudnn as cudnn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.GCN import JOINTNET_MASKNET_MEANSHIFT
from models.moe_modules import collect_moe_load_balance_loss, collect_moe_train_metrics
from datasets.skeleton_dataset import GraphDataset
from models.supplemental_layers.pytorch_chamfer_dist import chamfer_distance_with_average

import mlflow
import mlflow.pytorch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def pairwise_distances(x, y):
    #Input: x is a Nxd matrix
    #       y is an optional Mxd matirx
    #Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    #        if y is not given then use 'y=x'.
    #i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def meanshift_cluster(pts, bandwidth, weights, args):
    """
    meanshift written in pytorch
    :param pts: input points
    :param weights: weight per point during clustering
    :return: clustered points
    """
    pts_steps = []
    for i in range(args.meanshift_step):
        Y = pairwise_distances(pts, pts)
        K = torch.nn.functional.relu(bandwidth ** 2 - Y)
        if weights is not None:
            K = K * weights
        P = torch.nn.functional.normalize(K, p=1, dim=0, eps=1e-10)
        P = P.transpose(0, 1)
        pts = args.step_size * (torch.matmul(P, pts) - pts) + pts
        pts_steps.append(pts)
    return pts_steps


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
    model = JOINTNET_MASKNET_MEANSHIFT(
        use_moe_jointnet=args.use_moe_jointnet,
        use_moe_masknet=args.use_moe_masknet,
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_noise=args.router_noise,
    )
    model.to(device)

    optimizer = torch.optim.Adam([{'params': model.jointnet.parameters(), 'lr': args.jointnet_lr},
                                  {'params': model.masknet.parameters(), 'lr': args.masknet_lr},
                                  {'params': model.bandwidth, 'lr': args.bandwidth_lr}],
                                  weight_decay=args.weight_decay)
 
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
    else:
        pretrained_masknet = torch.load(args.masknet_resume)
        pretrained_jointnet = torch.load(args.jointnet_resume)
        model.masknet.load_state_dict(pretrained_masknet['state_dict'])
        model.jointnet.load_state_dict(pretrained_jointnet['state_dict'])

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(GraphDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=True, follow_batch=['joints'])
    val_loader = DataLoader(GraphDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
    test_loader = DataLoader(GraphDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False, follow_batch=['joints'])
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_metrics = test(test_loader, model, args, save_result=True, best_epoch=args.start_epoch)
        print('test_loss {:.8f}'.format(test_loss))
        print('MeanShift metrics:')
        for k, v in test_metrics.items():
            print('  {}: {:.6f}'.format(k, v))
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    logger = SummaryWriter(log_dir=args.logdir)

    mlflow.set_experiment("RigNet_Joint_Finetune")
    with mlflow.start_run(run_name=f"joint_finetune_{'moe_jointnet_' if args.use_moe_jointnet else ''}{'moe_masknet_' if args.use_moe_masknet else ''}{args.checkpoint.split('/')[-1]}"):
        log_args_to_mlflow(args)
        mlflow.log_param("device", str(device))

        for epoch in range(args.start_epoch, args.epochs):
            print('\nEpoch: %d ' % (epoch + 1))
            train_loss, moe_metrics = train(train_loader, model, optimizer, args)
            val_loss, val_metrics = test(val_loader, model, args)
            test_loss, test_metrics = test(test_loader, model, args)
            scheduler.step()
            print('Epoch{:d}. train_loss: {:.6f}.'.format(epoch + 1, train_loss))
            print('Epoch{:d}. val_loss: {:.6f}.'.format(epoch + 1, val_loss))
            print('Epoch{:d}. test_loss: {:.6f}.'.format(epoch + 1, test_loss))
            # print mean shift metrics
            print('  val_metrics: cd_after={:.6f}, avg_shift_last={:.6f}, total_disp={:.6f}, disp_std={:.6f}'.format(
                val_metrics['cd_after'], val_metrics['avg_shift_last'],
                val_metrics['total_disp'], val_metrics['disp_std']))
            print('  test_metrics: cd_after={:.6f}, avg_shift_last={:.6f}, total_disp={:.6f}, disp_std={:.6f}'.format(
                test_metrics['cd_after'], test_metrics['avg_shift_last'],
                test_metrics['total_disp'], test_metrics['disp_std']))

            # log scalar losses
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
                if "router_entropy" in moe_metrics:
                    mlflow.log_metric(
                        "train_moe_router_entropy",
                        moe_metrics["router_entropy"],
                        step=epoch + 1,
                    )
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
            mlflow.log_metric("test_loss", test_loss, step=epoch + 1)
            mlflow.log_metric("lr_jointnet", optimizer.param_groups[0]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_masknet", optimizer.param_groups[1]['lr'], step=epoch + 1)
            mlflow.log_metric("lr_bandwidth", optimizer.param_groups[2]['lr'], step=epoch + 1)

            # log mean shift metrics
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v), step=epoch + 1)
                logger.add_scalar(f"val/{k}", float(v), epoch + 1)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", float(v), step=epoch + 1)
                logger.add_scalar(f"test/{k}", float(v), epoch + 1)

            # remember best acc and save checkpoint
            is_best = val_loss < lowest_loss
            lowest_loss = min(val_loss, lowest_loss)
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss, 'optimizer': optimizer.state_dict()},
                            is_best, checkpoint=args.checkpoint)

            # tensorBoard basic loss
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
                if "router_entropy" in moe_metrics:
                    logger.add_scalar(
                        "train/moe_router_entropy",
                        moe_metrics["router_entropy"],
                        epoch + 1,
                    )
            logger.add_scalar("val/loss", val_loss, epoch + 1)
            logger.add_scalar("test/loss", test_loss, epoch + 1)

        best_model_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
        print("=> loading checkpoint '{}'".format(best_model_path))
        checkpoint = torch.load(best_model_path)
        best_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(best_model_path, best_epoch))
        test_loss, test_metrics = test(test_loader, model, args, save_result=True, best_epoch=best_epoch)
        print('Best epoch:\n test_loss {:.8f}'.format(test_loss))
        print('MeanShift metrics:')
        for k, v in test_metrics.items():
            print('  {}: {:.6f}'.format(k, v))


def train(train_loader, model, optimizer, args):
    global device
    model.train()  # switch to train mode
    loss_meter = AverageMeter()
    lb_loss_meter = AverageMeter()
    usage_spread_meter = AverageMeter()
    router_entropy_meter = AverageMeter()
    use_moe = args.use_moe_jointnet or args.use_moe_masknet
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        data_displacement, mask_pred_nosigmoid, mask_pred, bandwidth = model(data)
        y_pred = data_displacement + data.pos
        loss_total = 0.0
        for i in range(len(torch.unique(data.batch))):
            joint_gt = data.joints[data.joints_batch == i, :]
            y_pred_i = y_pred[data.batch == i, :]
            mask_pred_i = mask_pred[data.batch == i]
            loss_total += chamfer_distance_with_average(y_pred_i.unsqueeze(0), joint_gt.unsqueeze(0))
            clustered_pred = meanshift_cluster(y_pred_i, bandwidth, mask_pred_i, args)
            loss_ms = 0.0
            for j in range(args.meanshift_step):
                loss_ms += chamfer_distance_with_average(clustered_pred[j].unsqueeze(0), joint_gt.unsqueeze(0))
            loss_total = loss_total + args.ms_loss_weight * loss_ms / args.meanshift_step
        loss_total /= len(torch.unique(data.batch))
        if use_moe:
            moe_step_metrics = collect_moe_train_metrics(model)
            if 'load_balance_loss' in moe_step_metrics:
                lb_loss_meter.update(moe_step_metrics['load_balance_loss'])
            if 'expert_usage_spread' in moe_step_metrics:
                usage_spread_meter.update(moe_step_metrics['expert_usage_spread'])
            if 'router_entropy' in moe_step_metrics:
                router_entropy_meter.update(moe_step_metrics['router_entropy'])
            if args.moe_lb_weight > 0 and 'load_balance_loss' in moe_step_metrics:
                lb_loss = collect_moe_load_balance_loss(model)
                if not torch.is_tensor(lb_loss):
                    lb_loss = torch.tensor(lb_loss, device=device)
                loss_total = loss_total + args.moe_lb_weight * lb_loss
        if args.use_bce:
            mask_gt = data.mask.unsqueeze(1)
            loss_total += args.bce_loss_weight * torch.nn.functional.binary_cross_entropy_with_logits(mask_pred_nosigmoid, mask_gt.float(), reduction='mean')
        loss_total.backward()
        optimizer.step()
        loss_meter.update(loss_total.item())
    moe_metrics = {}
    if use_moe and lb_loss_meter.count > 0:
        moe_metrics['load_balance_loss'] = lb_loss_meter.avg
    if use_moe and usage_spread_meter.count > 0:
        moe_metrics['expert_usage_spread'] = usage_spread_meter.avg
    if use_moe and router_entropy_meter.count > 0:
        moe_metrics['router_entropy'] = router_entropy_meter.avg
    return loss_meter.avg, moe_metrics


def test(test_loader, model, args, save_result=False, best_epoch=None):
    """
    Evaluate model on test/val set and compute mean shift metrics.
    Returns:
        loss_avg: average loss
        metrics: dict with keys:
            'cd_after'         - Chamfer distance after mean shift (final)
            'avg_shift_last'   - average norm of shift between last two mean shift iterations
            'total_disp'       - average total displacement from initial to final
            'disp_std'         - standard deviation of total displacements
    """
    global device
    model.eval()
    loss_meter = AverageMeter()
    # meters for mean shift metrics
    cd_after_meter = AverageMeter()
    avg_shift_last_meter = AverageMeter()
    total_disp_meter = AverageMeter()
    disp_std_meter = AverageMeter()

    outdir = args.checkpoint.split('/')[-1]
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            data_displacement, mask_pred_nosigmoid, mask_pred, bandwidth = model(data)
            y_pred = data_displacement + data.pos
            loss_total = 0.0
            for i in range(len(torch.unique(data.batch))):
                joint_gt = data.joints[data.joints_batch == i, :]
                y_pred_i = y_pred[data.batch == i, :]
                mask_pred_i = mask_pred[data.batch == i]

                # chamfer before (for loss)
                cd_before = chamfer_distance_with_average(y_pred_i.unsqueeze(0), joint_gt.unsqueeze(0))
                loss_total += cd_before

                # run mean shift
                clustered_pred = meanshift_cluster(y_pred_i, bandwidth, mask_pred_i, args)
                y_pred_final = clustered_pred[-1]   # final positions

                # chamfer after mean shift
                cd_after = chamfer_distance_with_average(y_pred_final.unsqueeze(0), joint_gt.unsqueeze(0))
                cd_after_meter.update(cd_after.item())

                # mean shift dynamics metrics
                # 1. avg_shift_last: average norm of displacement between last two steps
                if len(clustered_pred) >= 2:
                    shift_last = torch.norm(clustered_pred[-1] - clustered_pred[-2], dim=1).mean().item()
                else:
                    shift_last = 0.0
                avg_shift_last_meter.update(shift_last)

                # 2. total_disp: average norm of total displacement (initial -> final)
                total_disp = torch.norm(y_pred_final - y_pred_i, dim=1).mean().item()
                total_disp_meter.update(total_disp)

                # 3. disp_std: std of total displacements
                disp_norms = torch.norm(y_pred_final - y_pred_i, dim=1)
                disp_std = disp_norms.std().item()
                disp_std_meter.update(disp_std)

                # loss from mean shift (as in original)
                loss_ms = 0.0
                for j in range(args.meanshift_step):
                    loss_ms += chamfer_distance_with_average(clustered_pred[j].unsqueeze(0), joint_gt.unsqueeze(0))
                loss_total = loss_total + args.ms_loss_weight * loss_ms / args.meanshift_step

                # cleanup
                del clustered_pred, y_pred_final
                torch.cuda.empty_cache()

                if save_result:
                    output_point_cloud_ply(y_pred_i, name=str(data.name[i].item()),
                                           output_folder='results/moe/{:s}/best_{:d}/'.format(outdir, best_epoch))
                    np.save('results/moe/{:s}/best_{:d}/{:d}_attn.npy'.format(outdir, best_epoch, data.name[i].item()),
                            mask_pred_i.data.to("cpu").numpy())
                    np.save('results/moe/{:s}/best_{:d}/{:d}_bandwidth.npy'.format(outdir, best_epoch, data.name[i].item()),
                            bandwidth.data.to("cpu").numpy())

            # average loss over graphs in batch
            loss_total /= len(torch.unique(data.batch))
            if args.use_bce:
                mask_gt = data.mask.unsqueeze(1)
                loss_total += args.bce_loss_weight * torch.nn.functional.binary_cross_entropy_with_logits(mask_pred_nosigmoid, mask_gt.float(), reduction='mean')
            loss_meter.update(loss_total.item())

    metrics = {
        'cd_after': cd_after_meter.avg,
        'avg_shift_last': avg_shift_last_meter.avg,
        'total_disp': total_disp_meter.avg,
        'disp_std': disp_std_meter.avg,
    }
    return loss_meter.avg, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyG DGCNN')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50], help='Decrease learning rate at these epochs.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on val/test set')
    parser.add_argument('--train_batch', default=1, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--test_batch', default=1, type=int, metavar='N', help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/moe/test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/test', type=str, metavar='LOG', help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_folder', default='/home/jovyan/data-storage/ModelResource_RigNetv1_preproccessed/train/', type=str, help='folder of training data')
    parser.add_argument('--val_folder', default='/home/jovyan/data-storage/ModelResource_RigNetv1_preproccessed/val/', type=str, help='folder of validation data')
    parser.add_argument('--test_folder', default='/home/jovyan/data-storage/ModelResource_RigNetv1_preproccessed/test/', type=str, help='folder of testing data')
    ######################
    parser.add_argument('--jointnet_lr', default=5e-5, type=float)
    parser.add_argument('--masknet_lr', default=5e-5, type=float)
    parser.add_argument('--bandwidth_lr', default=1e-6, type=float)
    parser.add_argument('--jointnet_resume', default='checkpoints/moe/pretrain_jointnet/model_best.pth.tar', type=str) # на данный момент pretrain_jointnetсохранено не в moe
    parser.add_argument('--masknet_resume', default='checkpoints/moe/pretrain_masknet/model_best.pth.tar', type=str)
    parser.add_argument('--meanshift_step', default=15, type=int, help='step size for meanshift update')
    parser.add_argument('--step_size', default=0.3, type=float)  # step size for meanshift
    parser.add_argument('--ms_loss_weight', default=2.0, type=float)  # weight for chamfer loss after meanshift
    parser.add_argument('--use_bce', action='store_true')  # if using mask supervision during finetuning
    parser.add_argument('--bce_loss_weight', default=0.1, type=float)  # weight for bce loss
    parser.add_argument('--no-use-moe-jointnet', dest='use_moe_jointnet', action='store_false', default=True, help='Disable MoE in JointNet and use the original MLP blocks')
    parser.add_argument('--no-use-moe-masknet', dest='use_moe_masknet', action='store_false', default=True, help='Disable MoE in MaskNet and use the original MLP blocks')    
    parser.add_argument('--num-experts', default=6, type=int, help='number of MoE experts')
    parser.add_argument('--top-k', default=1, type=int, help='top-k experts per token')
    parser.add_argument('--moe-lb-weight', default=0.001, type=float)
    parser.add_argument('--router-noise', default=0.01, type=float)
    parser.add_argument('--cuda', default=0, type=int, help='CUDA device index')

    args = parser.parse_args()
    print(args)
    main(args)
