import cv2
import ezdxf
import numpy as np
import logging
import pygicp

from ezdxf.document import Drawing
from ezdxf.math import Vec3
from ezdxf.entities.polyline import DXFVertex, Polyline
from tqdm import tqdm
from typing import Dict, List, Tuple, Iterator
# from RigidRegistration import RigidRegistrationnoScale as Registration


def iterate_over_entities(document: Drawing, layerlist: List[str] = list()) -> Iterator[Polyline]:
    for entity in tqdm(document.modelspace(), desc="Entities", leave=False):
        if entity.dxftype() != "POLYLINE":
            continue
        entity: Polyline = entity

        if layerlist and entity.dxf.layer not in layerlist:
            continue
        
        yield entity

def extract_json(
    document: Drawing,
    layerlist: List[str] = list(),
    remove_empty_layers: bool = True,
    remove_empty_vertices: bool = True
) -> Dict:
    """ Extract all `Polyline` entities from the document.
        Notes:
         - Modelspace iteration: https://ezdxf.readthedocs.io/en/stable/tutorials/getting_data.html#iterate-over-dxf-entities-of-a-layout.
         - Polyline: https://ezdxf.readthedocs.io/en/stable/dxfentities/polyline.html?highlight=Polyline.
         - Vertex: https://ezdxf.readthedocs.io/en/stable/dxfentities/polyline.html?highlight=Vertex#vertex.
    """
    layers: Dict[str, Dict] = {}
    mapping: Dict[str, str] = {}

    for index, layer in enumerate(document.layers.entries.values()):
        layers[f"layer_{index}"] = {
            "layer name": layer.dxf.name,
            "points": []
        }
        mapping[layer.dxf.name] = f"layer_{index}"

    for entity in iterate_over_entities(document, layerlist):
        points: List[Dict] = layers[mapping[entity.dxf.layer]]["points"]
        vertices: List[DXFVertex] = entity.vertices

        if remove_empty_vertices:
            vertices = [vertex for vertex in vertices if vertex.dxf.location != Vec3(0, 0, 0)]

        points.append({
            "point number": len(vertices),
            "coordinates": [coordinate for vertex in vertices for coordinate in vertex.dxf.location]
        })
    
    if remove_empty_layers:
        for layername, layer in list(layers.items()):
            if len(layer["points"]) == 0:
                del layers[layername]

    output: Dict[str, Dict] = {}
    output["header"] = {
        "layer number": len(layers.keys()),
        "structure number": [len(layer["points"]) for layer in layers.values()]
    }
    output.update(layers)
    return output

def extract_array(
    document: Drawing,
    normalize: bool = True,
    layerlist: List[str] = list()
) -> np.ndarray:
    vertices = []
    for entity in iterate_over_entities(document, layerlist):
        faces: List[List[DXFVertex]] = []
        if entity.is_poly_face_mesh:
            faces = entity.faces()       
        else:
            faces = [entity.vertices]

        for face in faces:
            for vertex in face:
                if vertex.dxf.location == Vec3(0, 0, 0):
                    continue
                x, y, z = vertex.dxf.location.xyz
                vertices.append([x, y, z])
    vertices = np.array(vertices, dtype=np.float32)

    if normalize:
        vertices = vertices / np.linalg.norm(vertices)
    return vertices

def extract(
    document: Drawing,
    normalize: bool = True,
    layerlist: List[str] = list()
) -> List[Dict]:
    xnorm, ynorm, znorm = [0, 0], [0, 0], [0, 0]

    structures: List = []
    for entity in iterate_over_entities(document, layerlist):
        description: Dict = {
            "layer": entity.dxf.layer,
            "faces": [],
        }

        faces: List[List[DXFVertex]] = []
        if entity.is_poly_face_mesh:
            faces = entity.faces()       
        else:
            faces = [entity.vertices]

        for face in faces:
            facepoints = []
            for vertex in face:
                if vertex.dxf.location == Vec3(0, 0, 0):
                    continue
                x, y, z = vertex.dxf.location.xyz
                xnorm = [min(xnorm[0], x), max(xnorm[1], x)]
                ynorm = [min(ynorm[0], y), max(ynorm[1], y)]
                znorm = [min(znorm[0], z), max(znorm[1], z)]
                facepoints.append([x, y, z])
            description["faces"].append(facepoints)
        structures.append(description)

    if normalize:
        logging.debug(f"Normalization (min, max): X {xnorm}, Y {ynorm}, Z {znorm}")
        norm = lambda a, limits: (a - limits[0]) / (limits[1] - limits[0])
        for structure in structures:
            for face in structure["faces"]:
                for index, (x, y, z) in enumerate(face):
                    x = norm(x, xnorm)
                    y = norm(y, ynorm)
                    z = norm(z, znorm)
                    face[index] = [x, y, z]

    return structures

def convert(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    xmat, ymat, zmat = np.degrees(matrix[:3, :3]).copy()

    # TODO: Ry, Rz
    # https://en.wikipedia.org/wiki/Rotation_matrix

    xmat[0, :] = 0
    xmat[:, 0] = 0
    xmat[0, 0] = 1
    xmat[1, 1] = np.cos(xmat[1, 1])
    xmat[1, 2] = -np.sin(xmat[1, 2])
    xmat[2, 1] = np.sin(xmat[2, 1])
    xmat[2, 2] = np.cos(xmat[2, 2])
    Rx = np.dot(xmat, np.array([vector[0], 0, 0]))

    ymat[:, 1] = 0
    ymat[1, :] = 0
    ymat[1, 1] = 1

    zmat[:, 2] = 0
    zmat[2, :] = 0
    zmat[2, 2] = 1
    return vector

def plot(
    structures: List[Dict],
    document: Drawing,
    width: int = 512, height: int = 512,
    monocolor: Tuple = None, origin: np.ndarray = None,
    matrix: np.ndarray = None) -> np.ndarray:
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

                # Vor the view only! Does not affect the model representation
                margin = -200
                x1, y1, z1 = x1 - margin, y1 - margin, z1 - margin
                x2, y2, z2 = x2 - margin, y2 - margin, z2 - margin
                cv2.line(origin, (x1, y1), (x2, y2), tuple(color), 2)
    return origin

def match(gtdoc: Drawing, tgdoc: Drawing, layerlist: List[str] = list()):
    normalize: bool = True

    logging.debug("Extracting structures from documents...")
    gtstruct = extract(gtdoc, normalize=normalize, layerlist=layerlist)
    tgstruct = extract(tgdoc, normalize=normalize, layerlist=layerlist)

    logging.debug("Extracting vertices matrices from documents...")
    gtmatrix = extract_array(gtdoc, normalize=normalize)
    tgmatrix = extract_array(tgdoc, normalize=normalize)

    logging.debug("Using FastGICP to create transition matrix...")
    gicp = pygicp.FastGICP()
    gicp.set_input_target(gtmatrix)
    gicp.set_input_source(tgmatrix)
    matrix = gicp.align()

    rotation = matrix[:3, :3]
    transform = matrix[:3, 3]
    matrix[:3, 3] = -np.dot(np.mean(tgmatrix, 0), rotation) + transform + np.mean(gtmatrix, 0)
    print("Matrix: \n", matrix)

    origin: np.ndarray = np.full((515 * 2, 512 * 2, 3), 255, dtype=np.uint8)
    origin = plot(gtstruct, gtdoc, 512, 512, (255, 0, 0), origin)
    origin = plot(tgstruct, tgdoc, 512, 512, (0, 0, 255), origin)

    cv2.imshow("Preview", origin)
    cv2.waitKey(0)
