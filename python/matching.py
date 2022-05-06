import cv2
import numpy as np
import logging

from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

import config
import loader
import utils

from calculations.cost_matrix import calculate_cost_matrix
from calculations.iou import iou_batch


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

def calculate_metrics(
    ground: np.ndarray,
    target: np.ndarray
) -> Dict:
    logging.debug(f"Counting metrics...")

    """ Returns linear assignment problem solution using scipy Hungarian implementation.
        Cost function between verticies is defined in `calculate_cost_matrix` method.
        Default: Euclidean distance (L2)."""
    logging.debug(f"Calculating cost matrix optimized: {config.enable_optimization}...")
    cost_matrix = calculate_cost_matrix(ground, target)

    # Cost matrix consists of ground (rows / height) and target (colls / width)
    height, width = cost_matrix.shape[:2]

    matches: Dict[List] = defaultdict(list)
    logging.debug("Applying linear_sum_assignment...")
    rows, cols = linear_sum_assignment(cost_matrix, maximize=False)
    for row, col in list(zip(rows, cols)):
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
    return dict(metrics)


def calculate_iou(
    gtindex: Dict[str, np.ndarray],
    gtendpoints: np.ndarray,
    tgindex: Dict[str, np.ndarray],
    tgendpoints: np.ndarray
) -> Dict[str, np.ndarray]:
    ious: Dict[str, List] = defaultdict(list)

    """ Calculate IoU based on GT classes only."""
    for key in gtindex.keys():
        gtsample = gtendpoints[gtindex[key]]
        tgsample = tgendpoints[tgindex.get(key, [])]

        logging.debug(f"Calculating 3D IoU for '{key}': {gtsample.shape[0]} over {tgsample.shape[0]} structures...")
        iou3d = iou_batch(gtsample, tgsample)[0]

        logging.debug("Applying linear_sum_assignment...")
        rows, cols = linear_sum_assignment(iou3d, maximize=True)
        for row, col in list(zip(rows, cols)):
            ious[key].append(iou3d[row, col])
        ious[key] = np.asarray(ious[key], dtype=np.float32)
        
    general = {}
    for classname, iou in ious.items():
        print("IoU: ", iou)
        general[classname] = {
            "min": iou.min(),
            "max": iou.max(),
            "mean": iou.mean(),
            "median": np.median(iou),
            "std": iou.std()
        }
    ious["general"] = general
    return ious

@utils.profile(output_root="profiling", enabled=config.debug)
def match(
    gtstructures: np.ndarray,
    tgstructures: np.ndarray
) -> Dict:
    logging.info("Matching models...")
    np.set_printoptions(precision=4, suppress=True)

    gtindex, gtendpoints = loader.read_endpoints(gtstructures)
    tgindex, tgendpoints = loader.read_endpoints(tgstructures)

    width, height = 1024, 1024
    origin: np.ndarray = np.full((width, height, 3), 255, dtype=np.uint8)
    origin = utils.plot(gtendpoints, gtstructures, width, height, monocolor=(255, 0, 0), origin=origin)
    origin = utils.plot(tgendpoints, tgstructures, width, height, monocolor=(0, 0, 255), origin=origin)

    """ Reshape to flatten arrays for point-cloud alignment."""
    ground = gtendpoints.reshape(-1, 3)
    logging.debug(f"Ground endpoints: mean {ground.mean():.2f}, max {ground.max(0)}, "
                    f"min {ground.min(0)}, size: {(ground.max(0) - ground.min(0))} "
                    f"(avg: {np.mean((ground.max(0) - ground.min(0))):.2f})")

    target = tgendpoints.reshape(-1, 3)
    logging.debug(f"Target endpoints: len  mean {target.mean():.2f}, max {target.max(0)}, "
                    f"min {target.min(0)}, size: {(target.max(0) - target.min(0))} "
                    f"(avg: {np.mean((target.max(0) - target.min(0))):.2f})")

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
    else:
        scale, rotation, translation = None, None, None

    """ Calculate base metrics: precision, recall."""
    """metrics = calculate_metrics(ground, target)
    for threshold, values in metrics.items():
        logging.info(f"Threshold: {threshold}")
        logging.info(f"  Metrics: {values}")"""

    """ Calculate 3D IoU grouped by classname: walls, collumns, doors."""
    ious = calculate_iou(gtindex, gtendpoints, tgindex, tgendpoints)
    for classname, iou in ious.items():
        if classname == "general":
            continue
        logging.info(f"Classname: {classname}")
        logging.info(f"  IoU: min {iou.min():.4f}, max {iou.max():.4f}, mean {iou.mean():.4f}, median {np.median(iou):.4f}, std {np.std(iou):.4f}")

    results = {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        # "metrics": metrics,
        "ious": ious,
    }

    if config.debug:
        results["config"] = {
            "enable_optimization": config.enable_optimization,
            "enable_normalization": config.enable_normalization,
            "metrics_thresholds": config.metrics_thresholds,
            "iou_thresholds": config.iou_thresholds,
            "units_multiplier": config.units_multiplier,
            "debug": config.debug
        }

    logging.debug("Showing preview data using OpenCV...")
    cv2.imshow("Preview (scaled)", origin)
    cv2.waitKey(0)
    return results
