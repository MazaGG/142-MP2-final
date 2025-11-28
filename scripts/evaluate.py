import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Haversine helper for geo exact distances
def haversine_dist_matrix(queries, points):
    q = np.radians(queries)
    p = np.radians(points)
    sin_q = np.sin((q[:, None, 0] - p[None, :, 0]) / 2.0)
    sin_lon = np.sin((q[:, None, 1] - p[None, :, 1]) / 2.0)
    a = sin_q**2 + np.cos(q[:, None, 0]) * np.cos(p[None, :, 0]) * sin_lon**2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  # Earth radius km
    return R * c

def brute_force_nn_geo(queries, points, k=10):
    D = haversine_dist_matrix(queries, points)
    inds = np.argsort(D, axis=1)[:, :k]
    return inds

def brute_force_nn_semantic(queries, points, k=10):
    D = cosine_distances(queries, points)
    inds = np.argsort(D, axis=1)[:, :k]
    return inds

def recall_at_k(gt_inds, approx_inds):
    Q, k = gt_inds.shape
    correct = 0
    for i in range(Q):
        correct += len(set(gt_inds[i]) & set(approx_inds[i]))
    return correct / (Q * k)
