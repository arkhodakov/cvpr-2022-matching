from calendar import c
import cv2
import numpy as np
import logging

from ezdxf.document import Drawing
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

import calculations
import config
import endpoints
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
    logging.debug("Calculating cost matrix using...")
    cost_matrix = cost_function(gt_normalized, tg_normalized)

    logging.debug("Applying linear_sum_assignment...")
    match_rows, match_colls = linear_sum_assignment(cost_matrix)
    return cost_matrix, match_rows, match_colls

def metrics(
    cost_matrix: np.ndarray,
    match_rows: List[int],
    match_colls: List[int]
) -> np.ndarray:
    logging.debug(f"Counting metrics...")
    threshold: float = np.max(config.accuracy_thresholds)
    matched_points: List = []
    matched, mismatched = 0, 0

    for row in match_rows:
        for col in match_colls:
            distance: float = cost_matrix[row, col]
            if distance < threshold:
                matched_points.append((row, col, distance))
                matched += 1
            else:
                mismatched += 1
    # TODO: Calculate `precision` and `recall`.
    logging.info(f"Matched: {matched}, mismatched: {mismatched}, accuracy: {(matched / (matched + mismatched) * 100):.2f}%")
    matched_points: np.ndarray = np.array(matched_points, dtype=[("gt", "int32"), ("tg", "int32"), ("distance", "float32")])
    return matched_points

def match(
    gtdoc: Drawing,
    tgdoc: Drawing,
    layerslist: List[str] = list()
) -> None:
    """ Notations: `gtdoc` - ground-truth .dxf document. `tgdoc` - target (user's prediction) .dxf document."""
    logging.info("Matching models...")
    np.set_printoptions(precision=2, suppress=True)

    logging.debug("Extracting vertices matrices from documents...")
    ground, gt_faces = endpoints.get_endpoints(gtdoc, layerslist=layerslist)
    logging.debug(f"Ground endpoints: mean {ground.mean():.2f}, max {ground.max(0)}, "
                    f"min {ground.min(0)}, size: {(ground.max(0) - ground.min(0))} "
                    f"(avg: {np.mean((ground.max(0) - ground.min(0))):.2f})")
    target, tg_faces = endpoints.get_endpoints(tgdoc, layerslist=layerslist)
    logging.debug(f"Target endpoints: mean {target.mean():.2f}, max {target.max(0)}, "
                    f"min {target.min(0)}, size: {(target.max(0) - target.min(0))} "
                    f"(avg: {np.mean((target.max(0) - target.min(0))):.2f})")
    
    width, height = 512, 512
    origin: np.ndarray = np.full((width, height, 3), 255, dtype=np.uint8)
    origin = utils.plot_endpoints(ground.copy(), gt_faces, width, height, monocolor=(255, 0, 0), origin=origin)
    origin = utils.plot_endpoints(target.copy(), tg_faces, width, height, monocolor=(0, 0, 255), origin=origin)

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

        origin = utils.plot_endpoints(target.copy(), tg_faces, width, height, monocolor=(0, 255, 0), origin=origin)

    """ Use Hungarian matching to find nearest points."""
    cost_matrix, match_rows, match_colls = lap(ground, target)

    matched_points = metrics(cost_matrix, match_rows, match_colls)
    # TODO: Save output data.

    logging.debug("Showing preview data using OpenCV...")
    cv2.imshow("Preview (scaled)", origin)
    cv2.waitKey(0)
