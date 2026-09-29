import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import List, Tuple, Union


def chamfer_distance_with_average(p1, p2):
    '''
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :return: sum of Chamfer Distance of two point sets

    Calculate Chamfer Distance between two point sets.
    '''
    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)
    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)
    p2 = p2.repeat(p1.size(0), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist_norm = torch.norm(dist, 2, dim=2)
    dist1 = torch.min(dist_norm, dim=1)[0] # for each point p1[i], the distance to the nearest point p2
    dist2 = torch.min(dist_norm, dim=0)[0] # for each point p2[j], the distance to the nearest point p1
    loss = 0.5 * ((torch.mean(dist1)) + (torch.mean(dist2)))
    return loss


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    :param p: shape (3,) or (N, 3)
    :param a: shape (3,) or (N, 3)
    :param b: shape (3,) or (N, 3)
    :return: scalar or array of shape (N,)

    Computes the distance from point p to segment [a, b].
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
    '''
    t = 0 - projection at point a;
    0 < t < 1 - ab;
    t = 1 - projection at point b;
    t < 0 - outside the segment on the side of point a;
    t > 1 - outside the segment on the side of point b.
    '''
    t = np.sum(ap * ab, axis=1, keepdims=True) / np.maximum(ab_dot_ab, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    
    closest = a + t * ab # parametric equation of the line segment: P(t)=a+t(b−a)
    dist = np.linalg.norm(p - closest, axis=1)
    return dist


def segment_to_segment_distance(a1: np.ndarray, b1: np.ndarray,
                                a2: np.ndarray, b2: np.ndarray) -> float:
    '''
    :param a1: shape (3,) — start of first segment
    :param b1: shape (3,) — end of first segment
    :param a2: shape (3,) — start of second segment
    :param b2: shape (3,) — end of second segment
    :return: float — minimum distance between two segments
    
    Computes the minimum distance between two segments [a1,b1] and [a2,b2].
    Each segment will be approximated by 50 points.
    This is not an analytical calculation of the distance between two segments, 
    but a numerical approximation through discretization.
    '''
    a1 = np.asarray(a1).flatten()
    b1 = np.asarray(b1).flatten()
    a2 = np.asarray(a2).flatten()
    b2 = np.asarray(b2).flatten()
    N = 50 # each segment will be approximated by 50 points
    t1 = np.linspace(0, 1, N)
    t2 = np.linspace(0, 1, N)

    # parametric equation of the line segment: P(t)=a+t(b−a)
    pts1 = a1[np.newaxis, :] + t1[:, np.newaxis] * (b1 - a1)[np.newaxis, :] # shape == (50, 3)
    pts2 = a2[np.newaxis, :] + t2[:, np.newaxis] * (b2 - a2)[np.newaxis, :] # shape == (50, 3)
    diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :] # shape == (50, 50, 3)
    dists = np.linalg.norm(diff, axis=2) # pairwise distance between all points of two segments
    return np.min(dists)


def sample_points_on_bones(joints: np.ndarray, edges: List[Tuple[int, int]],
                           num_samples_per_bone: int = 50) -> np.ndarray:
    '''
    :param joints: array (J, 3) of joint coordinates
    :param edges: list of pairs (i, j) — indices of joints connected by a bone
    :param num_samples_per_bone: number of points per bone
    :return: array (num_bones * num_samples_per_bone, 3)
    
    Discretizes each bone (segment) into uniformly distributed points.
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
    :param pred_joints: shape (N, 3) — predicted joints
    :param gt_joints: shape (M, 3) — ground-truth joints
    :return: float — CD-J2J value

    CD‑J2J (Chamfer Distance between Joints)
    '''
    pred = torch.tensor(pred_joints, dtype=torch.float32).unsqueeze(0)
    gt = torch.tensor(gt_joints, dtype=torch.float32).unsqueeze(0)
    return chamfer_distance_with_average(pred, gt).item()


def compute_cd_j2b_full(pred_joints: np.ndarray, pred_edges: List[Tuple[int, int]],
                        gt_joints: np.ndarray, gt_edges: List[Tuple[int, int]],
                        num_samples: int = 50) -> float:
    '''
    :param pred_joints: shape (N, 3) — predicted joints
    :param pred_edges: edges of predicted skeleton, indices in pred_joints
    :param gt_joints: shape (M, 3) — ground-truth joints
    :param gt_edges: edges of GT skeleton, indices in gt_joints
    :param num_samples: int — number of samples per bone
    :return: float — symmetric CD-J2B value

    Full CD‑J2B using discretization of predicted bones.
    For each predicted joint, find the nearest point on the GT bones; 
    for each GT joint, find the nearest point on the predicted bones; 
    symmetrically average the results.
    '''
    pred_bone_pts = sample_points_on_bones(pred_joints, pred_edges, num_samples)
    gt_bone_pts = sample_points_on_bones(gt_joints, gt_edges, num_samples)
    d1 = np.mean(np.min(cdist(pred_joints, gt_bone_pts), axis=1))
    d2 = np.mean(np.min(cdist(gt_joints, pred_bone_pts), axis=1))
    return 0.5 * (d1 + d2)


def compute_cd_b2b(pred_joints: np.ndarray, pred_edges: List[Tuple[int, int]],
                   gt_joints: np.ndarray, gt_edges: List[Tuple[int, int]]) -> float:
    '''
    :param pred_joints: shape (N, 3) — predicted joints
    :param pred_edges: edges of predicted skeleton, indices in pred_joints
    :param gt_joints: shape (M, 3) — ground-truth joints
    :param gt_edges: edges of GT skeleton, indices in gt_joints
    :return: float — symmetric CD-B2B value

    CD‑B2B (Chamfer Distance between Bones)
    The average distance from the predicted bones to the nearest GT bones, 
    then the reverse, and symmetrically average them.
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
    :param pred_joints: shape (N, 3) — predicted joints
    :param gt_joints: shape (M, 3) — ground-truth joints
    :param tolerance: float or np.ndarray or None — distance within which two joints are considered to coincide.
                      If None, computed as 0.05 * mean norm of gt_joints.
    :return: float — IoU value in [0, 1]
    
    IoU (Intersection over Union) for joints.
    '''
    if tolerance is None:
        scale = np.mean(np.linalg.norm(gt_joints, axis=1))
        tolerance = 0.05 * scale if scale > 0 else 0.01

    dist_matrix = cdist(pred_joints, gt_joints)
    row_ind, col_ind = linear_sum_assignment(dist_matrix) # searches for a correspondence between predicted joints and GT joints such that the total cost of the distances is minimized
    matched = 0 # intersection
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] < tolerance:
            matched += 1
    union_of_sets = len(pred_joints) + len(gt_joints) - matched
    if union_of_sets == 0:
        return 1.0
    return matched / union_of_sets


def compute_precision_recall(pred_joints: np.ndarray, gt_joints: np.ndarray,
                             tolerance: float) -> Tuple[float, float]:
    '''
    :param pred_joints: shape (N, 3) — predicted joints
    :param gt_joints: shape (M, 3) — ground-truth joints
    :param tolerance: float — distance threshold for matching
    :return: Tuple[float, float] — (precision, recall)

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
