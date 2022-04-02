import cv2
import numpy as np
import logging
import pygicp

from ezdxf.document import Drawing
from typing import List

import endpoints
import utils


def match(
    gtdoc: Drawing,
    tgdoc: Drawing,
    layerlist: List[str] = list(),
    apply_matrix: bool = True
) -> None:
    normalize: bool = True

    logging.debug("Extracting structures from documents...")
    gtstruct = endpoints.getStructures(gtdoc, normalize=normalize, layerlist=layerlist)
    tgstruct = endpoints.getStructures(tgdoc, normalize=normalize, layerlist=layerlist)

    logging.debug("Extracting vertices matrices from documents...")
    gtmatrix = endpoints.getEndpoints(gtdoc, normalize=normalize)
    tgmatrix = endpoints.getEndpoints(tgdoc, normalize=normalize)

    width, height = 512, 512
    origin: np.ndarray = np.full((width, height, 3), 255, dtype=np.uint8)
    origin = utils.plotStructures(gtstruct, gtdoc, width, height, (255, 0, 0), origin)

    if apply_matrix:
        logging.debug("Using FastGICP to create transition matrix...")
        gicp = pygicp.FastGICP()
        gicp.set_input_target(gtmatrix)
        gicp.set_input_source(tgmatrix)
        matrix = gicp.align()

        rotation = matrix[:3, :3]
        transform = matrix[:3, 3]
        matrix[:3, 3] = -np.dot(np.mean(tgmatrix, 0), rotation) + transform + np.mean(gtmatrix, 0)
        print("Matrix: \n", matrix)

        origin = utils.plotStructures(tgstruct, tgdoc, width, height, (0, 0, 255), origin, matrix=matrix)
    else:
        origin = utils.plotStructures(tgstruct, tgdoc, width, height, (0, 0, 255), origin)

    cv2.imshow("Preview", origin)
    cv2.waitKey(0)
