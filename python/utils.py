import cv2
import ezdxf
import numpy as np
import logging
import logging.config
import yaml

from ezdxf.document import Drawing
from tqdm import tqdm
from typing import Dict, List, Tuple


def load_logger():
    with open('logging.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

def plot_endpoints(
    vertices: np.ndarray,
    layers: np.ndarray,
    width: int = 512, height: int = 512, depth: int = 512,
    monocolor: Tuple = None, origin: np.ndarray = None
) -> np.ndarray:
    """ Plot all `Polyline` entities from the document."""
    scale: np.ndarray = np.array([width, height, depth], dtype=np.int32)
    margin: np.ndarray = (scale / 2).astype(np.int32)
    color = monocolor or (0, 0, 0)

    origin: np.ndarray = origin if origin is not None else np.full((height, width, 3), 255, dtype=np.uint8)
    faces = np.split(vertices, np.unique(layers[:, 0], return_index=True)[1])
    for face in tqdm(faces, desc="Plotting", leave=False):
        for i in range(0, len(face) - 1):
            pt1 = np.array(face[i] * scale + margin, dtype=np.int32)
            pt2 = np.array(face[i + 1] * scale + margin, dtype=np.int32)
            cv2.line(origin, tuple(pt1[:2]), tuple(pt2[:2]), tuple(color), 1)
    return origin

def plotStructures(
    structures: List[Dict],
    document: Drawing,
    width: int = 512, height: int = 512,
    monocolor: Tuple = None, origin: np.ndarray = None,
    matrix: np.ndarray = None,
    margin: int = 0) -> np.ndarray:
    """ Plot all `Polyline` entities from the document."""
    origin: np.ndarray = origin if origin is not None else np.full((height, width, 3), 255, dtype=np.uint8)
    for description in tqdm(structures, desc="Plotting", leave=False):
        layer = description["layer"]
        for face in description["faces"]:
            for i in range(0, len(face) - 1):
                x1, y1, z1 = face[i]
                x2, y2, z2 = face[i + 1]
                color = monocolor or ezdxf.colors.aci2rgb(document.layers.get(layer).color or 0)

                if matrix is not None:
                    x1, y1, z1 = np.dot(matrix[:3, :3], np.array([x1, y1, z1])) + matrix[:3, 3]
                    x2, y2, z2 = np.dot(matrix[:3, :3], np.array([x2, y2, z2])) + matrix[:3, 3]

                x1, y1, z1 = int(x1 * width), int(y1 * height), int(z1 * width)
                x2, y2, z2 = int(x2 * width), int(y2 * height), int(z2 * width)

                if margin > 0:
                    # Vor the view only! Does not affect the model representation
                    x1, y1, z1 = x1 - margin, y1 - margin, z1 - margin
                    x2, y2, z2 = x2 - margin, y2 - margin, z2 - margin
                cv2.line(origin, (x1, y1), (x2, y2), tuple(color), 2)
    return origin
