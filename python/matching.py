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
        print(f"[RigidRegistration] Matching iter: {iteration}, error: {error:.2f}", end="\rotation")

    callback = partial(print_iter)
    reg = RigidRegistration(**{'X': gt_normalized, 'Y': tg_normalized})
    # TODO: Speed up `register` with Numba.
    TY, (scale, rotation, translation) = reg.register(callback)
    return (scale, rotation, translation)

def lap(
    gt_normalized: np.ndarray,
    tg_normalized: np.ndarray,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """ Returns linear assignment problem solution using scipy Hungarian implementation.
        Cost function between verticies is defined in `calculate_cost_matrix` method.
        Default: Euclidean distance (L2)."""
    cost_function = calculations.calculate_cost_matrix_numba if config.enable_optimization else calculations.calculate_cost_matrix
    cost_matrix = cost_function(gt_normalized, tg_normalized)

    match_rows, match_colls = linear_sum_assignment(cost_matrix)
    return cost_matrix, match_rows, match_colls

def metrics(
    cost_matrix: np.ndarray,
    match_rows: List[int],
    match_colls: List[int]
) -> np.ndarray:
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
    layerslist: List[str] = list(),
    apply_matrix: bool = True
) -> None:
    """ Notations: `gtdoc` - ground-truth .dxf document. `tgdoc` - target (user's prediction) .dxf document."""
    np.set_printoptions(precision=4, suppress=True)
    normalize: bool = True

    logging.debug("Extracting vertices matrices from documents...")
    gt_source, gt_normalized, gt_faces = endpoints.get_endpoints(gtdoc, normalize=normalize, layerslist=layerslist, return_faces=True)
    logging.debug(f"GT Endpoints: mean {gt_normalized.mean(0)}, max {gt_normalized.max(0)}, min {gt_normalized.min(0)}")
    tg_source, tg_normalized, tg_faces = endpoints.get_endpoints(tgdoc, normalize=normalize, layerslist=layerslist, return_faces=True)
    logging.debug(f"TG Endpoints: mean {tg_normalized.mean(0)}, max {tg_normalized.max(0)}, min {tg_normalized.min(0)}")

    gt_vertices = gt_normalized if config.enable_normalization else gt_source
    tg_vertices = tg_normalized if config.enable_normalization else tg_source

    width, height = 512, 512
    origin: np.ndarray = np.full((width, height, 3), 255, dtype=np.uint8)
    origin = utils.plot_endpoints(gt_normalized, gt_faces, width, height, monocolor=(255, 0, 0), origin=origin)
    origin = utils.plot_endpoints(tg_normalized, tg_faces, width, height, monocolor=(0, 0, 255), origin=origin)

    if apply_matrix:
        """ Use Coherent Point Drift Algorithm for preprocessing alignment.
            Source: https://github.com/siavashk/pycpd."""
        scale, rotation, translation = align(gt_normalized, tg_normalized)

        """Matricies alignment formula:"""
        translation = -np.dot(np.mean(tg_normalized, 0), rotation) + translation + np.mean(gt_normalized, 0)
        tg_normalized = np.dot(tg_normalized, rotation) + translation
        tg_normalized *= scale

        origin = utils.plot_endpoints(tg_normalized, tg_faces, width, height, monocolor=(0, 255, 0), origin=origin)

    """ Use Hungarian matching to find nearest points."""
    cost_matrix, match_rows, match_colls = lap(gt_vertices, tg_vertices)

    matched_points = metrics(cost_matrix, match_rows, match_colls)
    print(matched_points)

    logging.debug("Showing preview data using OpenCV...")
    cv2.imshow("Preview", origin)
    cv2.waitKey(0)
