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


def _iterate_over_entities(document: Drawing, layerslist: List[str] = list()) -> Iterator[Polyline]:
    for entity in tqdm(document.modelspace(), desc="Entities", leave=False):
        if entity.dxftype() != "POLYLINE":
            continue
        entity: Polyline = entity

        if layerslist and entity.dxf.layer not in layerslist:
            continue
        
        yield entity

def exportJSON(
    document: Drawing,
    layerslist: List[str] = list(),
    remove_empty_layers: bool = True,
    remove_empty_vertices: bool = True
) -> Dict:
    """ Export all `Polyline` entities from the document.
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

    for entity in _iterate_over_entities(document, layerslist):
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

def get_endpoints(
    document: Drawing,
    normalize: bool = True,
    layerslist: List[str] = list(),
    return_faces: bool = False
) -> np.ndarray:
    vertices: List[List] = []
    layers: List[str] = []
    for entity in _iterate_over_entities(document, layerslist):
        faces: List[List[DXFVertex]] = []
        if entity.is_poly_face_mesh:
            faces = entity.faces()       
        else:
            faces = [entity.vertices]

        for index, face in enumerate(faces):
            for vertex in face:
                if vertex.dxf.location == Vec3(0, 0, 0):
                    continue
                vertices.append(vertex.dxf.location.xyz)
                layers.append([index, entity.dxf.layer])
    vertices: np.ndarray = np.array(vertices, dtype=np.float32)
    layers: np.ndarray = np.array(layers, dtype=np.object)

    if normalize:
        vertices = (vertices - np.min(vertices)) / (np.max(vertices) - np.min(vertices))
        vertices = vertices - vertices.mean(0)

    if return_faces:
        return vertices, layers
    else:
        return vertices

def get_structures(
    document: Drawing,
    normalize: bool = True,
    layerslist: List[str] = list()
) -> List[Dict]:
    xnorm, ynorm, znorm = [0, 0], [0, 0], [0, 0]

    structures: List = []
    for entity in _iterate_over_entities(document, layerslist):
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
        def norm (a, limits):
            # a -= ((limits[1] - limits[0]) / 2)
            a = (a - limits[0]) / (limits[1] - limits[0])
            return a

        for structure in structures:
            for face in structure["faces"]:
                for index, (x, y, z) in enumerate(face):
                    x = norm(x, xnorm)
                    y = norm(y, ynorm)
                    z = norm(z, znorm)
                    face[index] = [x, y, z]

    return structures
