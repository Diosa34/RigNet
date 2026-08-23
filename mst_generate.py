#-------------------------------------------------------------------------------
# Name:        mst_generate.py
# Purpose:     Generate skeleton as a tree based on predicted joints.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import os
import cv2
import argparse
import numpy as np
import open3d as o3d
from utils import binvox_rw
from utils.tree_utils import TreeNode
from utils.rig_parser import Skel
from utils.vis_utils import show_obj_skel, draw_shifted_pts
from utils.io_utils import readPly
from utils.cluster_utils import meanshift_cluster, nms_meanshift
from utils.mst_utils import primMST_symmetry, loadSkel_recur, increase_cost_for_outside_bone, flip, inside_check, sample_on_bone
from gen_dataset import get_geo_edges, get_tpl_edges
from geometric_proc.common_ops import calc_surface_geodesic

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from models.ROOT_GCN import ROOTNET
from models.PairCls_GCN import PairCls

import mlflow
from scipy.spatial.distance import cdist
from utils.log_utils import setup_device
from models.supplemental_layers.pytorch_chamfer_dist import (
    compute_cd_j2j,
    compute_cd_j2b_full,
    compute_cd_b2b,
    compute_iou,
    compute_precision_recall,
)


device = None


def predict_joints(model_id, args):
    """
    predict joints for a specified model
    :param model_id: processed model ID number
    :param args:
    :return: predicted joints, and voxelized mesh
    """
    vox_folder = os.path.join(args.dataset_folder, 'vox/')
    mesh_folder = os.path.join(args.dataset_folder, 'obj_remesh/')
    raw_pred = os.path.join(args.res_folder, '{:d}.ply'.format(model_id))
    vox_file = os.path.join(vox_folder, '{:d}.binvox'.format(model_id))
    mesh_file = os.path.join(mesh_folder, '{:d}.obj'.format(model_id))
    pred_attn = np.load(os.path.join(args.res_folder, '{:d}_attn.npy'.format(model_id)))

    with open(vox_file, 'rb') as fvox:
        vox = binvox_rw.read_as_3d_array(fvox)
    pred_joints = readPly(raw_pred)
    pred_joints, index_inside = inside_check(pred_joints, vox)
    pred_attn = pred_attn[index_inside, :]
    # img = draw_shifted_pts(mesh_file, pred_joints, weights=pred_attn)

    bandwidth = np.load(os.path.join(args.res_folder, '{:d}_bandwidth.npy'.format(model_id)))
    bandwidth = bandwidth[0]
    pred_joints = pred_joints[pred_attn.squeeze() > 1e-3]
    pred_attn = pred_attn[pred_attn.squeeze() > 1e-3]

    # reflect raw points
    pred_joints_reflect = pred_joints * np.array([[-1, 1, 1]])
    pred_joints = np.concatenate((pred_joints, pred_joints_reflect), axis=0)
    pred_attn = np.tile(pred_attn, (2, 1))
    # img = draw_shifted_pts(mesh_file, pred_joints, weights=pred_attn)
    # cv2.imwrite(os.path.join(res_folder, '{:s}_raw.jpg'.format(model_id)), img[:, :, ::-1])

    pred_joints = meanshift_cluster(pred_joints, bandwidth, pred_attn, max_iter=20)
    Y_dist = np.sum(((pred_joints[np.newaxis, ...] - pred_joints[:, np.newaxis, :]) ** 2), axis=2)
    density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
    # density = density * pred_attn
    density = np.sum(density, axis=0)
    density_sum = np.sum(density)
    pred_joints_ = pred_joints[density / density_sum > args.threshold_best]
    density_ = density[density / density_sum > args.threshold_best]
    pred_joints_ = nms_meanshift(pred_joints_, density_, bandwidth)
    pred_joints_, _ = flip(pred_joints_)

    reduce_threshold = args.threshold_best
    while len(pred_joints_) < 2 and reduce_threshold > 1e-7:
        # print('reducing')
        reduce_threshold = reduce_threshold / 1.3
        pred_joints_ = pred_joints[density / density_sum >= reduce_threshold]
        density_ = density[density / density_sum > reduce_threshold]
        pred_joints_ = nms_meanshift(pred_joints_, density_, bandwidth)
        pred_joints_, _ = flip(pred_joints_)
    if reduce_threshold <= 1e-7:
        pred_joints_ = nms_meanshift(pred_joints_, density, bandwidth)
        pred_joints_, _ = flip(pred_joints_)

    pred_joints = pred_joints_
    # img = draw_shifted_pts(mesh_file, pred_joints)
    # cv2.imwrite(os.path.join(res_folder, '{:d}_joint.jpg'.format(model_id)), img)
    # np.save(os.path.join(res_folder, '{:d}_joint.npy'.format(model_id)), pred_joints)
    return pred_joints, vox


def getInitId(data, model):
    """
    predict root joint ID via rootnet
    :param data:
    :param model:
    :return:
    """
    with torch.no_grad():
        root_prob, _ = model(data, shuffle=False)
        root_prob = torch.sigmoid(root_prob).data.cpu().numpy()
    root_id = np.argmax(root_prob)
    return root_id


def create_single_data(mesh, vox, surface_geodesic, pred_joints):
    """
    create data used as input to networks, wrapped by Data structure in pytorch-gemetric library
    :param mesh: input mesh loaded by open3d
    :param vox: voxelized mesh
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param pred_joints: predicted joints (numpy array, shape [J, 3])
    :return: wrapped data structure
    """
    mesh_v = np.asarray(mesh.vertices)
    mesh_vn = np.asarray(mesh.vertex_normals)
    mesh_f = np.asarray(mesh.triangles)

    # vertices and normals
    v = np.concatenate((mesh_v, mesh_vn), axis=1)
    v = torch.from_numpy(v).float()

    print("     gathering topological edges.")
    tpl_e = get_tpl_edges(mesh_v, mesh_f).T
    tpl_e = torch.from_numpy(tpl_e).long()
    tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))

    print("     gathering geodesic edges.")
    geo_e = get_geo_edges(surface_geodesic, mesh_v).T
    geo_e = torch.from_numpy(geo_e).long()
    geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))

    # batch for vertices (all zeros, because there is one object)
    batch = torch.zeros(len(v), dtype=torch.long)

    pair_all = []
    for joint1_id in range(len(pred_joints)):
        for joint2_id in range(joint1_id + 1, len(pred_joints)):
            dist = np.linalg.norm(pred_joints[joint1_id] - pred_joints[joint2_id])
            bone_samples = sample_on_bone(pred_joints[joint1_id], pred_joints[joint2_id])
            bone_samples_inside, _ = inside_check(bone_samples, vox)
            outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
            pair = np.array([joint1_id, joint2_id, dist, outside_proportion, 1])  # [i, j, dist, outside, label]
            pair_all.append(pair)
    pair_all = np.array(pair_all)
    pair_all_tensor = torch.from_numpy(pair_all).float()
    num_pair = len(pair_all)
    num_joint = len(pred_joints)

    # creating pair_attr (attributes for the model)
    # taking columns: [dist, outside_proportion, label] (indexes 2,3,4)
    pair_attr = pair_all_tensor[:, 2:]  # shape [num_pair, 3]

    # batch for joints and pairs (all zeros because one object)
    joints_batch = torch.zeros(num_joint, dtype=torch.long)
    pairs_batch = torch.zeros(num_pair, dtype=torch.long)

    # saving the original joints for the models
    joints_original = torch.from_numpy(pred_joints).float()

    # expand the joints to the mesh size for the y field (if necessary)
    if num_joint < len(mesh_v):
        pred_joints_expanded = np.tile(pred_joints, (int(np.ceil(len(mesh_v) / num_joint)), 1))
        pred_joints_expanded = pred_joints_expanded[:len(mesh_v), :]
    elif num_joint > len(mesh_v):
        pred_joints_expanded = pred_joints[:len(mesh_v), :]
    else:
        pred_joints_expanded = pred_joints
    pred_joints_expanded = torch.from_numpy(pred_joints_expanded).float()

    data = Data(
        x=torch.from_numpy(mesh_vn),
        pos=torch.from_numpy(mesh_v).float(),
        batch=batch,
        joints=joints_original,          # original joints
        y=pred_joints_expanded,          # extended (can be used in other places)
        pairs=pair_all_tensor,           # full information about pairs (indexes + attributes)
        pair_attr=pair_attr,             # attributes for the model (dist, outside, label)
        num_pair=[num_pair],
        tpl_edge_index=tpl_e,
        geo_edge_index=geo_e,
        num_joint=[num_joint],
        joints_batch=joints_batch,
        pairs_batch=pairs_batch
    ).to(device)
    return data


def run_mst_generate(args):
    """
    generate skeleton in batch
    :param args: input folder path and data folder path
    """
    global device
    device = setup_device(args.gpu)
    print('Using device: %s' % device)
    test_list = np.loadtxt(os.path.join(args.dataset_folder, 'test_final.txt'), dtype=int)
    root_select_model = ROOTNET()
    root_select_model.to(device)
    root_select_model.eval()
    root_checkpoint = torch.load(args.rootnet, map_location='cpu')
    root_select_model.load_state_dict(root_checkpoint['state_dict'])
    connectivity_model = PairCls()
    connectivity_model.to(device)
    connectivity_model.eval()
    conn_checkpoint = torch.load(args.bonenet, map_location='cpu')
    connectivity_model.load_state_dict(conn_checkpoint['state_dict'])

    # loading reference data from rig_info_remesh
    rig_info_folder = os.path.join(args.dataset_folder, 'rig_info_remesh/')

    all_cd_j2j = []
    all_cd_j2b = []
    all_cd_b2b = []
    all_iou = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_median_error = []
    all_pred_joint_count = []
    all_gt_joint_count = []
    all_ed = []

    for model_id in test_list:
        print(model_id)
        pred_joints, vox = predict_joints(model_id, args)
        mesh_filename = os.path.join(args.dataset_folder, 'obj_remesh/{:d}.obj'.format(model_id))
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        surface_geodesic = calc_surface_geodesic(mesh)
        data = create_single_data(mesh, vox, surface_geodesic, pred_joints)
        root_id = getInitId(data, root_select_model)
        with torch.no_grad():
            cost_matrix, _ = connectivity_model.forward(data)
            connect_prob = torch.sigmoid(cost_matrix)
        pair_idx = data.pairs.long().data.cpu().numpy()
        cost_matrix = np.zeros((data.num_joint[0], data.num_joint[0]))
        cost_matrix[pair_idx[:, 0], pair_idx[:, 1]] = connect_prob.data.cpu().numpy().squeeze()
        cost_matrix = cost_matrix + cost_matrix.transpose()
        cost_matrix = -np.log(cost_matrix+1e-10)
        cost_matrix = increase_cost_for_outside_bone(cost_matrix, pred_joints, vox)

        skel = Skel()
        parent, key, root_id = primMST_symmetry(cost_matrix, root_id, pred_joints)
        for i in range(len(parent)):
            if parent[i] == -1:
                skel.root = TreeNode('root', tuple(pred_joints[i]))
                break
        loadSkel_recur(skel.root, i, None, pred_joints, parent)
        try:
            img = show_obj_skel(mesh_filename, skel.root)
            cv2.imwrite(os.path.join(args.res_folder, '{:d}_skel.jpg'.format(model_id)), img[:,:,::-1])
        except Exception as e:
            print(f"Visualization failed for model {model_id}: {e}. Skipping image save.")
        skel.save(os.path.join(args.res_folder, '{:d}_skel.txt'.format(model_id)))

        # loading reference data from rig_info_remesh
        rig_file = os.path.join(rig_info_folder, '{:d}.txt'.format(model_id))
        if not os.path.exists(rig_file):
            print(f"Rig info file not found for model {model_id}, skipping metrics.")
            continue

        gt_joints = []
        gt_parent = []
        gt_root_name = None
        joint_name_to_idx = {}
        with open(rig_file, 'r') as f:
            lines = f.readlines()
        print(f"  Read {len(lines)} lines from {rig_file}")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0].lower() in ['joint', 'joints']:
                # format: joint(s) name x y z
                name = parts[1]
                pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                joint_name_to_idx[name] = len(gt_joints)
                gt_joints.append(pos)
                gt_parent.append(-1)  # temporarily, it will be filled in later from hier
            elif parts[0].lower() == 'root':
                gt_root_name = parts[1]
            elif parts[0].lower() in ['hier', 'bone', 'edge', 'parent']:
                # hier parent child
                parent_name = parts[1]
                child_name = parts[2]
                if parent_name in joint_name_to_idx and child_name in joint_name_to_idx:
                    parent_idx = joint_name_to_idx[parent_name]
                    child_idx = joint_name_to_idx[child_name]
                    gt_parent[child_idx] = parent_idx
                else:
                    print(f"  Warning: unknown joint names in hier: {parent_name}, {child_name}")
            else:
                # if the string contains two joint names without a keyword (for flexibility)
                if len(parts) == 2 and parts[0] in joint_name_to_idx and parts[1] in joint_name_to_idx:
                    parent_idx = joint_name_to_idx[parts[0]]
                    child_idx = joint_name_to_idx[parts[1]]
                    gt_parent[child_idx] = parent_idx

        gt_joints = np.array(gt_joints)
        if gt_root_name is not None and gt_root_name in joint_name_to_idx:
            gt_root = joint_name_to_idx[gt_root_name]
        else:
            # if root is not specified, we search for the root (parent == -1)
            gt_root = None
            for i, p in enumerate(gt_parent):
                if p == -1:
                    gt_root = i
                    break

        # building edges from gt_parent
        gt_edges = []
        for child, par in enumerate(gt_parent):
            if par != -1:
                gt_edges.append((par, child))

        print(f"  Found {len(gt_joints)} joints, {len(gt_edges)} edges, root={gt_root}")

        if len(gt_edges) == 0:
            print(f"Model {model_id}: no edges in ground truth, skipping metrics.")
            continue

        # generating predicted edges from parent
        pred_edges = []
        for child, par in enumerate(parent):
            if par != -1:
                pred_edges.append((par, child))

        if len(pred_edges) == 0:
            print(f"Skipping metrics for {model_id} due to empty predicted edges.")
            continue

        # tolerance for IoU/Precision/Recall (scale is the average distance to the center)
        scale = np.mean(np.linalg.norm(gt_joints, axis=1))
        tolerance = 0.05 * scale if scale > 0 else 0.01

        cd_j2j = compute_cd_j2j(pred_joints, gt_joints)
        all_cd_j2j.append(cd_j2j)

        cd_j2b = compute_cd_j2b_full(pred_joints, pred_edges, gt_joints, gt_edges)
        all_cd_j2b.append(cd_j2b)

        cd_b2b = compute_cd_b2b(pred_joints, pred_edges, gt_joints, gt_edges)
        all_cd_b2b.append(cd_b2b)

        iou = compute_iou(pred_joints, gt_joints, tolerance)
        all_iou.append(iou)

        precision, recall = compute_precision_recall(pred_joints, gt_joints, tolerance)
        all_precision.append(precision)
        all_recall.append(recall)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        all_f1.append(f1)

        dist_matrix = cdist(gt_joints, pred_joints)
        min_gt_to_pred = dist_matrix.min(axis=1)
        all_median_error.append(np.median(min_gt_to_pred))
        all_pred_joint_count.append(len(pred_joints))
        all_gt_joint_count.append(len(gt_joints))

        # Tree Edit Distance (ED) with apted
        if gt_root is not None:
            try:
                from apted import APTED
                from apted.helpers import Tree as AptedTree
        
                def convert_to_apted(node):
                    # all nodes have the same label to count only the structure
                    apted_node = AptedTree('0')
                    if hasattr(node, 'children'):
                        for child in node.children:
                            apted_node.children.append(convert_to_apted(child))
                    return apted_node
        
                # building a reference tree from gt_edges
                gt_children = [[] for _ in range(len(gt_joints))]
                for a, b in gt_edges:
                    gt_children[a].append(b)
        
                def build_tree_from_edges(joints, edges, root_idx):
                    children = [[] for _ in range(len(joints))]
                    for a, b in edges:
                        children[a].append(b)
                    def recurse(idx):
                        node = TreeNode(f'joint_{idx}', tuple(joints[idx]))
                        for child_idx in children[idx]:
                            node.children.append(recurse(child_idx))
                        return node
                    return recurse(root_idx)
        
                gt_tree = build_tree_from_edges(gt_joints, gt_edges, gt_root)
        
                # building the predicted tree from parent
                pred_children = [[] for _ in range(len(pred_joints))]
                for child, par in enumerate(parent):
                    if par != -1:
                        pred_children[par].append(child)
                pred_root_idx = None
                for i, p in enumerate(parent):
                    if p == -1:
                        pred_root_idx = i
                        break
                if pred_root_idx is None:
                    print(f"  No root in predicted skeleton for model {model_id}")
                    continue
                pred_tree = build_tree_from_edges(pred_joints, pred_edges, pred_root_idx)
        
                apted_pred = convert_to_apted(pred_tree)
                apted_gt = convert_to_apted(gt_tree)
        
                ted = APTED(apted_pred, apted_gt).compute_edit_distance()
                all_ed.append(ted)
            except Exception as e:
                print(f"ED computation failed for {model_id}: {e}")

    mlflow.set_experiment("RigNet_mst_evaluation")
    with mlflow.start_run(run_name="mst_generation"):
        mlflow.log_param("dataset_folder", args.dataset_folder)
        mlflow.log_param("res_folder", args.res_folder)
        mlflow.log_param("rootnet", args.rootnet)
        mlflow.log_param("bonenet", args.bonenet)
        mlflow.log_param("threshold_best", args.threshold_best)

        if all_cd_j2j:
            mlflow.log_metric("test_CD_J2J", np.mean(all_cd_j2j))
        if all_cd_j2b:
            mlflow.log_metric("test_CD_J2B", np.mean(all_cd_j2b))
        if all_cd_b2b:
            mlflow.log_metric("test_CD_B2B", np.mean(all_cd_b2b))
        if all_iou:
            mlflow.log_metric("test_IoU", np.mean(all_iou))
        if all_precision:
            mlflow.log_metric("test_Precision", np.mean(all_precision))
        if all_recall:
            mlflow.log_metric("test_Recall", np.mean(all_recall))
        if all_f1:
            mlflow.log_metric("test_F1", np.mean(all_f1))
        if all_median_error:
            mlflow.log_metric("test_MedianJointError", np.mean(all_median_error))
        if all_pred_joint_count:
            mlflow.log_metric("test_PredJointCount", np.mean(all_pred_joint_count))
        if all_gt_joint_count:
            mlflow.log_metric("test_GTJointCount", np.mean(all_gt_joint_count))
        if all_ed:
            mlflow.log_metric("test_TreeEditDist", np.mean(all_ed))

        print("Evaluation metrics logged to MLflow.")
        print(f"Average CD-J2J: {np.mean(all_cd_j2j):.6f}, CD-J2B: {np.mean(all_cd_j2b):.6f}, CD-B2B: {np.mean(all_cd_b2b):.6f}")
        print(f"Average IoU: {np.mean(all_iou):.6f}, Precision: {np.mean(all_precision):.6f}, Recall: {np.mean(all_recall):.6f}, F1: {np.mean(all_f1):.6f}, MedianError: {np.mean(all_median_error):.6f}, ED: {np.mean(all_ed):.6f}")
        print(f"Avg pred joints: {np.mean(all_pred_joint_count):.2f}, avg GT joints: {np.mean(all_gt_joint_count):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_folder', default='/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/', type=str)
    parser.add_argument('--res_folder', default='results/gcn_meanshift/best_25/', type=str)
    parser.add_argument('--rootnet', default='checkpoints/rootnet/model_best.pth.tar', type=str)
    parser.add_argument('--bonenet', default='checkpoints/bonenet/model_best.pth.tar', type=str)
    parser.add_argument('--threshold_best', default=1e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int,
                        help='CUDA device index, e.g. 0 for cuda:0, 2 for cuda:2 (default: 0)')
    args = parser.parse_args()
    print(args)
    run_mst_generate(args)