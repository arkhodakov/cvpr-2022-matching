import numpy as np
import numba as nb

from numba import njit, prange
from scipy.optimize import linear_sum_assignment

@njit
def calculate_cost_matrix_numba(
    gt_normalized: nb.float32[:, :],
    tg_normalized: nb.float32[:, :]
) -> nb.float32[:, :]:
    # Performance tested: 15x faster ~ 1s.
    costs = np.ones((gt_normalized.shape[0], tg_normalized.shape[0]), np.float32)
    for i in prange(gt_normalized.shape[0]):
        for j in prange(tg_normalized.shape[0]):
            costs[i, j] = np.sqrt(np.power(gt_normalized[i] - tg_normalized[j], 2))[0]
    return costs

def calculate_cost_matrix(
    gt_normalized: np.ndarray,
    tg_normalized: np.ndarray,
) -> np.ndarray:
    costs = np.full((gt_normalized.shape[0], tg_normalized.shape[0]), np.nan, dtype=np.float32)
    for i in range(gt_normalized.shape[0]):
        for j in range(tg_normalized.shape[0]):
            costs[i, j] = np.sqrt(np.power(gt_normalized[i] - tg_normalized[j], 2))[0]
    return costs
