import cv2
import numpy as np
import logging

from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

import calculations
import config
import loader
import iou
import utils


def align(gt_normalized: np.ndarray, tg_normalized: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """ Returns alignment `scale` factor, `rotation` matrix and `translation` matrix to transform
        target matrix `tg_normalized` to `gt_normalized`."""
    from functools import partial
    from pycpd import RigidRegistration

    logging.debug("Using RigidRegistration implementation to calculate the matrix...")
    def print_iter(iteration, error, X, Y):
        print(f"[RigidRegistration] Matching iter: {iteration}, error: {error:.2f}", end="\r")

    callback = partial(print_iter)
    reg = RigidRegistration(**{'X': gt_normalized, 'Y': tg_normalized})
    # TODO: Speed up `register` with Numba.
    # TODO: !IMPORTANT Find a measurable error of the alignment (`q`, ...).
    TY, (scale, rotation, translation) = reg.register(callback)
    print()
    return (scale, rotation, translation)

def lap(
    gt_normalized: np.ndarray,
    tg_normalized: np.ndarray,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """ Returns linear assignment problem solution using scipy Hungarian implementation.
        Cost function between verticies is defined in `calculate_cost_matrix` method.
        Default: Euclidean distance (L2)."""
    cost_function = calculations.calculate_cost_matrix_numba if config.enable_optimization else calculations.calculate_cost_matrix
    logging.debug(f"Calculating cost matrix optimized: {config.enable_optimization}...")
    cost_matrix = cost_function(gt_normalized, tg_normalized)

    logging.debug("Applying linear_sum_assignment...")
    match_rows, match_colls = linear_sum_assignment(cost_matrix)
    return cost_matrix, match_rows, match_colls

def calculate_metrics(
    cost_matrix: np.ndarray,
    match_rows: List[int],
    match_colls: List[int]
) -> Tuple[np.ndarray, Dict]:
    logging.debug(f"Counting metrics...")
    matches: Dict[List] = defaultdict(list)

    # Cost matrix consists of ground (rows / height) and target (colls / width)
    height, width = cost_matrix.shape[:2]

    for i in range(len(match_rows)):
        row, col = match_rows[i], match_colls[i]
        distance: float = cost_matrix[row, col]
        for threshold in config.metrics_thresholds:
            if distance < threshold:
                matches[threshold].append(distance)
    metrics: Dict = defaultdict(dict)

    for threshold, matched in matches.items():
        # Calculate metrics according to `compute_precision_recall_helper`:
        # https://github.com/seravee08/WarpingError_Floorplan/blob/main/IOU_precision_recall/ipynb/main.ipynb
        precision: float = len(matched) / width
        recall: float = len(matched) / height
        f1: float = (2 * precision * recall) / (precision + recall)
        metrics[threshold] = {
            "matched": len(matched),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    return metrics

def calculate_iou(
    gtendpoints: np.ndarray,
    tgendpoints: np.ndarray
):
    return iou.box3d_iou(gtendpoints[120], tgendpoints[120])

def match(
    gtstructures: np.ndarray,
    tgstructures: np.ndarray
) -> None:
    """ Notations: `gtdoc` - ground-truth .dxf document. `tgdoc` - target (user's prediction) .dxf document."""
    logging.info("Matching models...")
    np.set_printoptions(precision=4, suppress=True)

    """logging.debug(f"Ground endpoints: mean {ground.mean():.2f}, max {ground.max(0)}, "
                    f"min {ground.min(0)}, size: {(ground.max(0) - ground.min(0))} "
                    f"(avg: {np.mean((ground.max(0) - ground.min(0))):.2f})")
    target, tg_faces = endpoints.get_endpoints(tgdoc, layerslist=layerslist)
    logging.debug(f"Target endpoints: len  mean {target.mean():.2f}, max {target.max(0)}, "
                    f"min {target.min(0)}, size: {(target.max(0) - target.min(0))} "
                    f"(avg: {np.mean((target.max(0) - target.min(0))):.2f})")"""
    
    gtendpoints = loader.read_endpoints(gtstructures)
    tgendpoints = loader.read_endpoints(tgstructures)

    width, height = 780, 780
    origin: np.ndarray = np.full((width, height, 3), 255, dtype=np.uint8)
    origin = utils.plot_endpoints(gtendpoints, width, height, monocolor=(255, 0, 0), origin=origin)
    origin = utils.plot_endpoints(tgendpoints, width, height, monocolor=(0, 0, 255), origin=origin)

    ground = gtendpoints.reshape(-1, 3)
    target = tgendpoints.reshape(-1, 3)

    if config.enable_normalization:
        """ Use Coherent Point Drift Algorithm for preprocessing alignment.
            Source: https://github.com/siavashk/pycpd."""
        scale, rotation, translation = align(ground, target)
        logging.info(f"Alignment scale ratio: {scale:.8f}")

        """Matricies alignment formula:"""
        translation = -np.dot(np.mean(target, 0), rotation) + translation + np.mean(ground, 0)
        target = np.dot(target, rotation) + translation
        target *= scale

        logging.debug(f"Aligned target endpoints: mean {target.mean():.2f}, max {target.max(0)}, "
            f"min {target.min(0)}, size: {(target.max(0) - target.min(0))} "
            f"(avg: {np.mean((target.max(0) - target.min(0))):.2f})")
        origin = utils.plot_endpoints(target, width, height, monocolor=(0, 255, 0), origin=origin)

    """ Use Hungarian matching to find nearest points."""
    # cost_matrix, match_rows, match_colls = lap(ground, target)

    # metrics = calculate_metrics(cost_matrix, match_rows, match_colls)
    # print("Metrics: ", metrics)

    iou = calculate_iou(gtendpoints, tgendpoints)
    print("IoU: ", iou)

    logging.debug("Showing preview data using OpenCV...")
    """cv2.imshow("Preview (scaled)", origin)
    cv2.waitKey(0)"""
