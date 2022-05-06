import cv2
import numpy as np

from typing import Tuple

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1


def plot(
    endpoints: np.ndarray,
    structures: np.ndarray,
    width: int = 780, height: int = 780, depth: int = 780,
    monocolor: Tuple = None, origin: np.ndarray = None
) -> np.ndarray:
    """ Plot stuctures from the endpoints."""
    endpoints = endpoints.reshape(-1, 3)
    endpoints = (endpoints - endpoints.min()) / (endpoints.max() - endpoints.min())
    endpoints = endpoints - endpoints.mean(0)
    endpoints = endpoints / endpoints.max()
    endpoints = endpoints.reshape(-1, 8, 3)

    centers = structures[:, :6].reshape(-1, 3)
    centers = (centers - centers.min()) / (centers.max() - centers.min())
    centers = centers - centers.mean(0)
    centers = centers / centers.max()
    structures[:, :6] = centers.reshape(-1, 6)

    padding: float = 0.8
    scale: np.ndarray = np.array([width, height, depth], dtype=np.int32) * padding / 2
    margin: np.ndarray = (np.array([width, height, depth]) / 2).astype(np.int32)

    color = monocolor or (0, 0, 0)

    origin: np.ndarray = origin if origin is not None else np.full((height, width, 3), 255, dtype=np.uint8)
    for index in range(endpoints.shape[0]):
        # Plot bbox endpoints.
        coordinates = endpoints[index, :4]
        coordinates = coordinates * scale + margin
        tl, tr, bl, br = coordinates[:, :2].astype(int)

        cv2.line(origin, tl, tr, (128, 128, 128), 1)
        cv2.line(origin, bl, br, (255, 0, 0), 1)
        cv2.line(origin, tl, bl, (0, 256, 0), 1)
        cv2.line(origin, br, tr, (0, 0, 256), 1)

        # Plot middle line.
        (x1, y1, _) = (structures[index, :3] * scale + margin).astype(int)
        (x2, y2, _) = (structures[index, 3:6] * scale + margin).astype(int)
        # drawline(origin, (x1, y1), (x2, y2), (0, 0, 0), 1, gap=5)
        """cv2.imshow("Preview", origin)
        cv2.waitKey(0)"""
    return origin