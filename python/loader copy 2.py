import json
import cv2
import numpy as np

from pathlib import Path
from typing import Dict, List, Tuple


def read_endpoints(structures: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """ box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
                5 ----------- 6
               /|            /|        z
              / |           / |        |  y
             /  |          /  |        | /
            4 --1-------- 7 --2        |/
            |  /          |  /         0-------x
            | / p1 --- p2 | /
            |/            |/
            0 ----------- 3
    """
    index = []
    endpoints = []
    for structure in structures:
        x1, y1, z1 = structure[:3]
        x2, y2, z2 = structure[3:6]
        
        width, depth, height = structure[6:9]
        w2, d2, h2 = (width / 2) + 0.25, (depth / 2) + 0.25, (height / 2)

        angle: float = np.arctan2((y2 - y1), (x2 - x1)) # Source points rotation OX (radians)
        angle = np.rad2deg(angle)

        rotation: float = structure[10] # Structure specicic rotation (degrees)
        angle += rotation # Final rotation angle in degrees
        angle -= 45

        sides = np.array([
            [-w2, +d2],
            [+w2, +d2],
            [-w2, -d2],
            [+w2, -d2]
        ], dtype=np.float32)
        sides = np.tile(sides, (2, 1))

        """ 2D [x,y] vector rotation matrix.
            Docs: https://matthew-brett.github.io/teaching/rotation_2d.html."""
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        for i in range(sides.shape[0]):
            sides[i] = np.dot(R, sides[i])

        # Points:      [0 , 1 , 2 -- , 3 -- , 4 -- , 5 -- , 6 -- , 7 -- ]
        x = np.asarray([x1, x1, x2, x2, x1   , x1   , x2   , x2   ])
        y = np.asarray([y1, y1, y2, y2, y1   , y1   , y2   , y2   ])
        z = np.asarray([z1, z1, z1, z1, z1+h2, z1+h2, z1+h2, z1+h2])

        corners = np.vstack([x, y, z]).transpose()
        corners[:, :2] += sides
        endpoints.append(corners)

        classname = structure[9]
        index.append(classname)
    endpoints = np.asarray(endpoints, dtype=np.float32)

    index = np.asarray(index, dtype=str)
    
    unique, indices = np.unique(index, return_index=True)
    index = dict(zip(unique, np.split(np.arange(len(index)), indices[1:])))
    return index, endpoints


def read_structures(data: Dict) -> np.ndarray:
    """ Encapsulate all classes data into general Structured Numpy array:
        `(x1, y1, z1, x2, y2, z2, width, depth, height, classname: str)`."""
    structureslist: List = []
    for classname, structures in data.items():
        for structure in structures:
            x1, y1, z1 = structure.get("start_pt", structure.get("loc", [.0, .0, .0]))
            x2, y2, z2 = structure.get("end_pt", structure.get("loc", [.0, .0, .0]))
            width = structure.get("width", .0)
            depth = structure.get("depth", width)
            height = structure.get("height", 0)
            rotation = structure.get("rotation", 0)
            structureslist.append([
                x1, y1, z1,
                x2, y2, z2,
                width, depth, height,
                classname, rotation
            ])
    return np.array(structureslist, dtype=object)


def read_json(path: Path) -> Dict:
    """Read data from separate JSON data files."""

    """ Identificator - unique structure name. For example:
        01_OfficeLab01_Allfloors - identificator,
        _columns - classname."""
    assert path.suffix == ".json", "JSON files only parsing is available."
    identificator = "_".join(path.name.split("_")[:-1])

    data: Dict = {}
    files = list(path.parent.glob(f"{identificator}*.json"))
    for file in files:
        classname = file.name.split("_")[-1].split(".")[0]
        with open(file, "r", encoding="utf-8") as buffer:
            data[classname] = json.load(buffer)
    return data
