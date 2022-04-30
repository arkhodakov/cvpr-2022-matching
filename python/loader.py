import json
import numpy as np

from pathlib import Path
from typing import Dict, List


def read_endpoints(structures: np.ndarray) -> np.ndarray:
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
    endpoints = []
    for structure in structures:
        x1, y1, z1 = structure[:3]
        x2, y2, z2 = structure[3:6]
        
        width, depth, height = structure[6:9]
        w2, d2, h2 = (width / 2), (depth / 2), (height / 2)

        d = np.array([(x2 - x1), (y2 - y1)], dtype=np.float32)
        if np.sum(d) == 0:
            """Set `d` equals to 1 in case we have only one location: doors class"""
            d = [1.0, 1.0]
        else:
            d /= np.linalg.norm(d + np.finfo(np.float32).eps)

        dx = -d[1] * w2
        dy = d[0] * d2

        # Points:   [0 -- , 1 -- , 2 -- , 3 -- , 4 -- , 5 -- , 6 -- , 7 -- ]
        x_corners = [x1-dx, x1-dx, x2+dx, x2+dx, x1-dx, x1+dx, x2+dx, x2-dx]
        y_corners = [y1-dy, y1+dy, y2-dy, y2+dy, y1-dy, y1+dy, y2-dy, y2+dy]
        z_corners = [z1   , z1   , z1   , z1   , z1+h2, z1+h2, z1+h2, z1+h2]

        corners = np.vstack([x_corners,y_corners,z_corners]).transpose()
        corners = corners
        endpoints.append(corners)
    endpoints = np.asarray(endpoints, dtype=np.float32)
    return endpoints


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
            structureslist.append([
                x1, y1, z1,
                x2, y2, z2,
                width, depth, height,
                classname
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
