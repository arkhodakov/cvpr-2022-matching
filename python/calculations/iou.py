import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from typing import Tuple


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
        subjectPolygon: a list of (x,y) 2d points, any polygon.
        clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
        **points have to be counter-clockwise ordered**

    Return:
        a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) >= (cp2[1]-cp1[1])*(p[0]-cp1[0])
    
    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]] 
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
    
    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
    
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]
    
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    # 1D implementation: 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return 0.5 * np.abs(np.sum(x * np.roll(y, 1, axis=-1), axis=-1) - np.sum(y * np.roll(x, 1, axis=-1), axis=-1))

def convex_hull_intersection(p1, p2):
    result = np.zeros((p1.shape[0], p2.shape[0]))
    for i in range(len(p1)):
        for j in range(len(p2)):
            inter_p = polygon_clip(p1[i], p2[j])
            if inter_p is not None:
                hull_inter = ConvexHull(inter_p)
                result[i, j] = hull_inter.volume
    return result

def counter_clockwise(rectangle: np.ndarray) -> np.ndarray:
    x, y = rectangle[..., 0], rectangle[..., 1]
    filter = np.sum(x * np.roll(y, 1, axis=-1), axis=-1) - np.sum(y * np.roll(x, 1, axis=-1), axis=-1) > 0
    rectangle[filter] = rectangle[filter][:, ::-1]
    return rectangle

def volume(corners) -> float:
    """ corners: (8,3) no assumption on axis direction """
    print(f"Calculating volume [{corners.shape}]: \n...")
    width = np.sqrt(np.sum(np.power(corners[..., 0, :2] - corners[..., 1, :2], 2), axis=1))
    print(" Width: ", width)
    length = np.sqrt(np.sum(np.power(corners[..., 1, :2] - corners[..., 2, :2], 2), axis=1))
    print(" Length: ", length)
    height = np.abs(corners[..., 0, 2] - corners[..., 4, 2])
    print(" Height: ", height)
    return (width * length * height)

def iou_batch(
    ground: np.ndarray,
    target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute 3D bounding box IoU.

    Parameters:
        ground: numpy array (n,8,3), assume up direction is Z
        target: numpy array (n,8,3), assume up direction is Z
    Return:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    """
    minimal = np.min([np.min([ground, target]), .0])
    ground -= minimal
    target -= minimal
    print("Minimal: ", minimal)

    print(f"Identical: {np.array_equal(ground, target)}")

    gface = ground[:, :4][..., :2]
    print(f"Ground Face [{gface.shape}, {gface.dtype}]: \n", gface[:1], "\n...")
    tface = target[:, :4][..., :2]
    print(f"Target Face [{tface.shape}, {tface.dtype}]: \n", tface[:1], "\n...")
    
    gface = counter_clockwise(gface)
    print(f"Rect1 [{gface.shape}, {gface.dtype}].")
    tface = counter_clockwise(tface)
    print(f"Rect2 [{tface.shape}, {tface.dtype}].")
    
    garea = poly_area(gface[..., 0], gface[..., 1])
    print(f"Area1 [{garea.shape}, {garea.dtype}]: \n", garea)
    tarea = poly_area(tface[..., 0], tface[..., 1])
    print(f"Area2 [{tarea.shape}, {tarea.dtype}]: \n", tarea)

    inter_area = convex_hull_intersection(gface, tface)
    print(f"Inter Area [{inter_area.shape}, {inter_area.dtype}]: \n", inter_area)

    intersected_filter = np.argwhere(np.any(inter_area, axis=1)).ravel()
    print("Intersected Filter: \n", intersected_filter)

    inter_area = inter_area[intersected_filter]
    garea = garea[intersected_filter]
    tarea = tarea[intersected_filter]

    iou_2d = inter_area / (np.tile(garea, (inter_area.shape[0], 1)) + np.tile(tarea, (inter_area.shape[0], 1)) - inter_area)
    print("2D IoU: \n", iou_2d)

    ground_filtered = ground[intersected_filter]
    target_filtered = target[intersected_filter]

    zmax = np.min([ground_filtered[..., 4, 2], target_filtered[..., 4, 2]], axis=0)
    zmin = np.max([ground_filtered[..., 0, 2], target_filtered[..., 0, 2]], axis=0)
    difference = np.max([np.full_like(zmax, .0), zmax - zmin], axis=0)
    print(f"Difference [{difference.shape}, {difference.dtype}]: \n", difference)

    inter_vol = inter_area * difference
    print("Inter Volume: \n", inter_vol)

    vol1 = volume(ground_filtered)
    print("Ground Volume: \n", vol1)
    vol2 = volume(target_filtered)
    print("Target Volume: \n", vol2)

    iou_3d = np.zeros((ground.shape[0], target.shape[0]), dtype=np.float32)
    iou_3d[intersected_filter] = inter_vol / (np.tile(vol1, (inter_vol.shape[0], 1)) + np.tile(vol2, (inter_vol.shape[0], 1)) - inter_vol)
    print("3D IoU: \n", iou_3d)
    return iou_3d, iou_2d
