import cv2
import numpy as np
import logging

from ezdxf.document import Drawing
from typing import List, Tuple

import config
import endpoints
import utils

def align_cpp(gtmatrix: np.ndarray, tgmatrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError()

def align_python(gtmatrix: np.ndarray, tgmatrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from functools import partial
    from RigidRegistration import RigidRegistrationnoScale as Registration

    logging.debug("Using RigidRegistration implementation to calculate the matrix...")
    def print_iter(iteration, error, X, Y):
        print(f"[RigidRegistration] Matching iter: {iteration}, error: {error:.2f}", end="\r")

    callback = partial(print_iter)
    reg = Registration(**{'X': gtmatrix, 'Y': tgmatrix})
    TY, (R, t) = reg.register(callback)
    return (R, t)

def match(
    gtdoc: Drawing,
    tgdoc: Drawing,
    layerslist: List[str] = list(),
    apply_matrix: bool = True
) -> None:
    np.set_printoptions(precision=4, suppress=True)
    normalize: bool = True

    logging.debug("Extracting vertices matrices from documents...")
    gtmatrix, gtfaces = endpoints.getEndpoints(gtdoc, normalize=normalize, layerslist=layerslist, return_faces=True)
    logging.debug(f"GT Endpoints: mean {gtmatrix.mean(0)}, max {gtmatrix.max(0)}, min {gtmatrix.min(0)}")
    tgmatrix, tgfaces = endpoints.getEndpoints(tgdoc, normalize=normalize, layerslist=layerslist, return_faces=True)
    logging.debug(f"TG Endpoints: mean {tgmatrix.mean(0)}, max {tgmatrix.max(0)}, min {tgmatrix.min(0)}")

    width, height = 512, 512
    origin: np.ndarray = np.full((width, height, 3), 255, dtype=np.uint8)
    origin = utils.plotEndpoints(gtmatrix, gtfaces, width, height, monocolor=(255, 0, 0), origin=origin)

    if apply_matrix:
        transformed = tgmatrix.copy()
        if config.use_cpp_matching:
            R, t = align_python(gtmatrix, tgmatrix)
        else:
            R, t = align_python(gtmatrix, tgmatrix)

        t = -np.dot(np.mean(transformed, 0), R) + t + np.mean(gtmatrix, 0)
        transformed = np.dot(transformed, R) + t

        origin = utils.plotEndpoints(transformed, tgfaces, width, height, monocolor=(0, 255, 0), origin=origin)

    origin = utils.plotEndpoints(tgmatrix, tgfaces, width, height, monocolor=(0, 0, 255), origin=origin)

    logging.debug("Showing preview data using OpenCV...")
    cv2.imshow("Preview", origin)
    cv2.waitKey(0)
