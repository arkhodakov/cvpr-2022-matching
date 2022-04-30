import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


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
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def get_angle(a, b, c):
   angle = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
   return angle + 360 if angle < 0 else angle

def is_convex(points):
   n = len(points)
   for i in range(len(points)):
      p1 = points[i-2]
      p2 = points[i-1]
      p3 = points[i]
      if get_angle(p1, p2, p3) > 180:
         return False
   return True


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[4,0] - corners[6,0])**2))
    b = np.sqrt(np.sum((corners[2,1] - corners[3,1])**2))
    c = np.sqrt(np.sum((corners[0,2] - corners[4,2])**2))
    return a*b*c


def iou_box(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    minimal = np.min([np.min(corners1), np.min(corners2)])
    if minimal < 0:
        corners1 -= minimal
        corners2 -= minimal
    
    indices = [2, 0, 1, 3] # [2, 0, 1, 3]
    rect1 = corners1[indices][:, :2]
    rect2 = corners2[indices][:, :2]
    # print("Rectangles: ", rect1, rect2)

    area1 = poly_area(rect1[:, 0], rect1[:, 1])
    area2 = poly_area(rect2[:, 0], rect2[:, 1])
    # print("Areas: ", area1, area2)

    if is_clockwise(rect1):
        rect1 = rect1[::-1]
    if is_clockwise(rect2):
        rect2 = rect2[::-1]

    inter_area = convex_hull_intersection(rect1, rect2)[1]
    # print("Inter area: ", inter_area)
    iou_2d = inter_area/(area1+area2-inter_area)

    zmax = min(corners1[4, 2], corners2[4, 2])
    zmin = max(corners1[0, 2], corners2[0, 2])
    inter_vol = inter_area * max(0.0, zmax-zmin)
    # print("Inter Vol: ", inter_vol)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    # print("Vols: ", vol1, vol2)
    iou = abs(inter_vol / (vol1 + vol2 - inter_vol))
    return iou, iou_2d

def box3d_vol_batch(corners):
    ''' corners: (n,8,3) no assumption on axis direction '''
    l = np.sqrt(np.linalg.norm(corners[:, 1, :] - corners[:, 2, :], axis=1))
    w = np.sqrt(np.linalg.norm(corners[:, 0, :] - corners[:, 1, :], axis=1))
    h = np.sqrt(np.linalg.norm(corners[:, 0, :] - corners[:, 4, :], axis=1))
    return l*w*h

def iou_batch(batch_corners1, batch_corners2):
    '''
    Input:
        batch_corners1: numpy array (n,8,3), assume up direction is negative Y
        batch_corners2: numpy array (m,8,3), assume up direction is negative Y
    Output:
        batch_iou: 3D bounding box IoU (n,m)
    '''
    n = batch_corners1.shape[0]
    m = batch_corners2.shape[1] #suppose m < n

    vol_batch1 = box3d_vol_batch(batch_corners1) #n
    vol_batch2 = box3d_vol_batch(batch_corners2) #m

    y_max_batch1 = batch_corners1[:,0,1] #n
    y_min_batch1 = batch_corners1[:,4,1] #n
    y_max_batch2 = batch_corners2[:,0,1] #m
    y_min_batch2 = batch_corners2[:,4,1] #m

    batch_iou = np.zeros((n,m), dtype=np.float32)
    for i in range(m):
        rect2 = batch_corners2[i][[3, 1, 0, 2]][::-1, :2]
        vol2 = vol_batch2[i]

        y_max = np.where(y_max_batch1-y_max_batch2[i]<0, y_max_batch1, y_max_batch2[i]) #n
        y_min = np.where(y_min_batch1-y_min_batch2[i]>0, y_min_batch1, y_min_batch2[i]) #n
        inter_y = np.where(y_max-y_min < 0., 0., y_max-y_min) #n
        inter_area = np.zeros((n), dtype=np.float32) #n
        for j in range(n):
            rect1 = batch_corners1[j][[3, 1, 0, 2]][::-1, :2]
            inter_area[j] = convex_hull_intersection(rect1, rect2)[1]
        inter_vol = inter_y * inter_area #n
        batch_iou[:,i] = inter_vol/(vol_batch1+vol2-inter_vol)

    return batch_iou
