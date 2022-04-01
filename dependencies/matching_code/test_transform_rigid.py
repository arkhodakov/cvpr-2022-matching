import numpy as np
import open3d as o3d 
from jsonfile import load_json_to_points
from lazfile import load_laz_to_points
import os
import copy
import matplotlib.pyplot as plt
from functools import partial
import json
import glob


def print_iter(iteration, error, X, Y):
    print("iter: ", iteration, "error: ", error)

def draw_points(X, Y):
    plt.cla()
    plt.scatter(X[:, 0],  X[:, 1], s=0.5, color='red', label='Target')
    plt.scatter(Y[:, 0],  Y[:, 1], s=0.5, color='blue', label='Source')
    plt.legend(loc='upper left', fontsize='x-large')
    plt.draw()

def compute_point_to_floorplan(base_path, building_name, R, t, split = "test"):
    json_path = os.path.join(base_path, "json_" + split, building_name + "_floorplan.txt")
    json_points = load_json_to_points(json_path)
    num_points_json = json_points.shape[0]

    points_o3d_path = os.path.join(base_path, "points_ply_" + split, building_name + ".ply")
    pcd_points = o3d.io.read_point_cloud(points_o3d_path)
    laz_points = np.array(pcd_points.points)

    #transformed_laz = np.dot(laz_points[:, :2], R) + t
    #laz_points[:, :2] = transformed_laz

    transformed_laz = np.dot(json_points[:, :2], R) + t
    json_points[:, :2] = transformed_laz
    draw_points(json_points, laz_points)

    import ipdb; ipdb.set_trace()

    png_path = os.path.join(base_path, 'matching_png_noscale_' + split, building_name + '.png')
    plt.savefig(png_path, dpi=300)

    json_dict = {'R': R.tolist(), 't': t.tolist()}
    point_to_floorplan =  os.path.join(base_path, 'point_to_floorplan_noscale_' + split, building_name + '.json')
    with open(point_to_floorplan, 'w') as f:
        json.dump(json_dict, f)

if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    base_path = "./"


    building_name = '25_Parking_01_F3'

    R = np.array([[0.933288647497005,	-0.359127136702554],	[0.359127136702554,	0.933288647497005]])
    #R = np.array([[1.0, 0], [0, 1.0]])
    t = np.array([40.5042225945809, 18.5971221907152])

    compute_point_to_floorplan(base_path, building_name, R, t, split = 'test')


    



