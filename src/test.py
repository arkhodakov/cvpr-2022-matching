import cv2
import numpy as np

x, y = center = [120, 120]
w, d = [10, 10]

for i in range(0, 360, 10):
    # TODO: Draw point and calculate rotation
    origin = np.full((256, 256, 3), 255, dtype=np.uint8)

    b = np.deg2rad(i)
    mat = np.array([
        [np.cos(b), -np.sin(b)],
        [np.sin(b), np.cos(b)]
    ])

    """ Default rotation and position.
        Left side:
            x - width
            y +- depth
        Right side:
            x + width
            y +- depth
    """
    sides = np.array([
        [-w, +d],
        [+w, +d],
        [-w, -d],
        [+w, -d]
    ])
    for i in range(sides.shape[0]):
        sides[i] = np.dot(mat, sides[i])

    coordinates = np.vstack([np.tile(x, 4), np.tile(y, 4)]).transpose()
    coordinates += sides

    tl, tr, bl, br = coordinates.astype(int)

    cv2.circle(origin, center, 2, (0, 0, 255), -1)

    cv2.line(origin, tl, tr, (56, 56, 56), 1)
    cv2.line(origin, bl, br, (128, 128, 128), 1)
    cv2.line(origin, tl, bl, (0, 255, 0), 1)
    cv2.line(origin, br, tr, (0, 0, 255), 1)

    cv2.imshow("Preview", origin)
    cv2.waitKey(0)
