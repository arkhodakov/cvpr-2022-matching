import cv2
import numpy as np
import logging
import logging.config
import yaml

from typing import Tuple


def load_logger():
    with open('logging.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

def plot_structures(
    endpoints: np.ndarray,
    width: int = 780, height: int = 780, depth: int = 780,
    monocolor: Tuple = None, origin: np.ndarray = None
) -> np.ndarray:
    """ Plot stuctures from the endpoints."""
    padding: float = 0.8
    scale: np.ndarray = np.array([width, height, depth], dtype=np.int32) * padding / 2
    margin: np.ndarray = (np.array([width, height, depth]) / 2).astype(np.int32)

    color = monocolor or (0, 0, 0)

    origin: np.ndarray = origin if origin is not None else np.full((height, width, 3), 255, dtype=np.uint8)
    for structure in endpoints:
        x1, y1, z1 = structure[:3] * scale + margin
        x2, y2, z2 = structure[3:6] * scale + margin

        pt1 = np.asarray([x1, y1, z1], dtype=np.int32)
        pt2 = np.asarray([x2, y2, z2], dtype=np.int32)
        cv2.line(origin, tuple(pt1[:2]), tuple(pt2[:2]), tuple(color), 1)

        width, depth, height = structure[6:9] * scale
        w2, d2, h2 = (width / 2), (depth / 2), (height / 2)
        
        dxy = np.array([(x2 - x1), (y2 - y1)], dtype=np.float32)
        dxy = dxy / np.linalg.norm(dxy) 

        dx = -dxy[1]
        dy = dxy[0]

        tl = (int(x2 - dx * w2), int(y2 - dy * d2))
        tr = (int(x2 + dx * w2), int(y2 + dy * d2))
        bl = (int(x1 - dx * w2), int(y1 - dy * d2))
        br = (int(x1 + dx * w2), int(y1 + dy * d2))

        cv2.line(origin, tl, tr, (0, 255, 0), 1)
        cv2.line(origin, bl, br, (0, 255, 0), 1)
        cv2.line(origin, tl, bl, (0, 255, 0), 1)
        cv2.line(origin, br, tr, (0, 255, 0), 1)
    return origin


def plot_endpoints(
    endpoints: np.ndarray,
    width: int = 780, height: int = 780, depth: int = 780,
    monocolor: Tuple = None, origin: np.ndarray = None
) -> np.ndarray:
    """ Plot stuctures from the endpoints."""
    endpoints = endpoints.reshape(-1, 3)
    endpoints = (endpoints - endpoints.min()) / (endpoints.max() - endpoints.min())
    endpoints = endpoints - endpoints.mean(0)
    endpoints = endpoints / endpoints.max()
    endpoints = endpoints.reshape(-1, 4, 3)

    padding: float = 0.8
    scale: np.ndarray = np.array([width, height, depth], dtype=np.int32) * padding / 2
    margin: np.ndarray = (np.array([width, height, depth]) / 2).astype(np.int32)

    color = monocolor or (0, 0, 0)

    origin: np.ndarray = origin if origin is not None else np.full((height, width, 3), 255, dtype=np.uint8)
    for endpoint in endpoints:
        coordinates = endpoint[:4]
        coordinates = coordinates * scale + margin
        tl, tr, bl, br = coordinates[:, :2].astype(int)

        cv2.line(origin, tl, tr, color, 1)
        cv2.line(origin, bl, br, color, 1)
        cv2.line(origin, tl, bl, color, 1)
        cv2.line(origin, br, tr, color, 1)
    return origin