import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import List, Tuple, Union


def chamfer_distance_with_average(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''
    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)
    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)
    p2 = p2.repeat(p1.size(0), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist_norm = torch.norm(dist, 2, dim=2)
    dist1 = torch.min(dist_norm, dim=1)[0]
    dist2 = torch.min(dist_norm, dim=0)[0]
    loss = 0.5 * ((torch.mean(dist1)) + (torch.mean(dist2)))
    return loss


#  Functions for working with segments (bones)

def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Computes the distance from point p to segment [a, b].
    All arguments are arrays of shape (3,) or (N, 3).
    Returns a scalar or array of shape (N,).
    '''
    p = np.asarray(p)
    a = np.asarray(a)
    b = np.asarray(b)

    if p.ndim == 1:
        p = p.reshape(1, -1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    ab = b - a
    ap = p - a
    ab_dot_ab = np.sum(ab * ab, axis=1, keepdims=True)
    t = np.sum(ap * ab, axis=1, keepdims=True) / np.maximum(ab_dot_ab, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    dist = np.linalg.norm(p - closest, axis=1)
    return dist


def segment_to_segment_distance(a1: np.ndarray, b1: np.ndarray,
                                a2: np.ndarray, b2: np.ndarray) -> float:
    '''
    Computes the minimum distance between two segments [a1,b1] and [a2,b2].
    All arguments are arrays of shape (3,).
    '''
    a1 = np.asarray(a1).flatten()
    b1 = np.asarray(b1).flatten()
    a2 = np.asarray(a2).flatten()
    b2 = np.asarray(b2).flatten()
    N = 50
    t1 = np.linspace(0, 1, N)
    t2 = np.linspace(0, 1, N)
    pts1 = a1[np.newaxis, :] + t1[:, np.newaxis] * (b1 - a1)[np.newaxis, :]
    pts2 = a2[np.newaxis, :] + t2[:, np.newaxis] * (b2 - a2)[np.newaxis, :]
    diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return np.min(dists)


def sample_points_on_bones(joints: np.ndarray, edges: List[Tuple[int, int]],
                           num_samples_per_bone: int = 50) -> np.ndarray:
    '''
    Discretizes each bone (segment) into uniformly distributed points.
    :param joints: array (J, 3) of joint coordinates
    :param edges: list of pairs (i, j) — indices of joints connected by a bone
    :param num_samples_per_bone: number of points per bone
    :return: array (num_bones * num_samples_per_bone, 3)
    '''
    pts_list = []
    for i, j in edges:
        a = joints[i]
        b = joints[j]
        for t in np.linspace(0, 1, num_samples_per_bone):
            pt = a + t * (b - a)
            pts_list.append(pt)
    return np.array(pts_list)


#  Metrics for evaluating the skeleton (described in section 6 of the article)

def compute_cd_j2j(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
    '''
    CD‑J2J (Chamfer Distance between Joints)
    '''
    pred = torch.tensor(pred_joints, dtype=torch.float32).unsqueeze(0)
    gt = torch.tensor(gt_joints, dtype=torch.float32).unsqueeze(0)
    return chamfer_distance_with_average(pred, gt).item()


def compute_cd_j2b_full(pred_joints: np.ndarray, pred_edges: List[Tuple[int, int]],
                        gt_joints: np.ndarray, gt_edges: List[Tuple[int, int]],
                        num_samples: int = 50) -> float:
    '''
    Full CD‑J2B using discretization of predicted bones.
    '''
    pred_bone_pts = sample_points_on_bones(pred_joints, pred_edges, num_samples)
    gt_bone_pts = sample_points_on_bones(gt_joints, gt_edges, num_samples)
    d1 = np.mean(np.min(cdist(pred_joints, gt_bone_pts), axis=1))
    d2 = np.mean(np.min(cdist(gt_joints, pred_bone_pts), axis=1))
    return 0.5 * (d1 + d2)


def compute_cd_b2b(pred_joints: np.ndarray, pred_edges: List[Tuple[int, int]],
                   gt_joints: np.ndarray, gt_edges: List[Tuple[int, int]]) -> float:
    '''
    CD‑B2B (Chamfer Distance between Bones)
    '''
    dists_pred_to_gt = []
    for i1, j1 in pred_edges:
        a1, b1 = pred_joints[i1], pred_joints[j1]
        min_d = np.inf
        for i2, j2 in gt_edges:
            a2, b2 = gt_joints[i2], gt_joints[j2]
            d = segment_to_segment_distance(a1, b1, a2, b2)
            if d < min_d:
                min_d = d
        dists_pred_to_gt.append(min_d)
    d1 = np.mean(dists_pred_to_gt) if dists_pred_to_gt else 0.0

    dists_gt_to_pred = []
    for i2, j2 in gt_edges:
        a2, b2 = gt_joints[i2], gt_joints[j2]
        min_d = np.inf
        for i1, j1 in pred_edges:
            a1, b1 = pred_joints[i1], pred_joints[j1]
            d = segment_to_segment_distance(a2, b2, a1, b1)
            if d < min_d:
                min_d = d
        dists_gt_to_pred.append(min_d)
    d2 = np.mean(dists_gt_to_pred) if dists_gt_to_pred else 0.0

    return 0.5 * (d1 + d2)


def compute_iou(pred_joints: np.ndarray, gt_joints: np.ndarray,
                tolerance: Union[float, np.ndarray] = None) -> float:
    '''
    IoU (Intersection over Union) for joints.
    '''
    if tolerance is None:
        scale = np.mean(np.linalg.norm(gt_joints, axis=1))
        tolerance = 0.05 * scale if scale > 0 else 0.01

    dist_matrix = cdist(pred_joints, gt_joints)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matched = 0
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] < tolerance:
            matched += 1
    denominator = len(pred_joints) + len(gt_joints) - matched
    if denominator == 0:
        return 1.0
    return matched / denominator


def compute_precision_recall(pred_joints: np.ndarray, gt_joints: np.ndarray,
                             tolerance: float) -> Tuple[float, float]:
    '''
    Precision and Recall for joints.
    '''
    dist_matrix = cdist(pred_joints, gt_joints)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] < tolerance:
            tp += 1
    fp = len(pred_joints) - tp
    fn = len(gt_joints) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def compute_tree_edit_distance(pred_root, gt_root) -> int:
    '''
    ED (Tree Edit Distance) using the zss library.
    Works with TreeNode from utils.tree_utils (has children and name attributes)[reference:1].
    '''
    try:
        import zss
    except ImportError:
        raise ImportError("Для вычисления Tree Edit Distance установите библиотеку zss: pip install zss")

    def get_children(node):
        return node.children if hasattr(node, 'children') else []

    def get_label(node):
        return node.name if hasattr(node, 'name') else str(node)

    return zss.distance(pred_root, gt_root, get_children, get_label)