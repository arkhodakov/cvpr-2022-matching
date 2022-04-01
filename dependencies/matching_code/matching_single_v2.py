import numpy as np
import open3d as o3d 
from jsonfile import load_json_to_points
from lazfile import load_laz_to_points
import os
import copy
import matplotlib.pyplot as plt
from functools import partial
#from pycpd import RigidRegistrationnoScale as Registration
from RigidRegistration import RigidRegistrationnoScale as Registration
import json
import glob

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def print_iter(iteration, error, X, Y):
    print("iter: ", iteration, "error: ", error)

def draw_points(X, Y):
    plt.cla()
    plt.scatter(X[:, 0],  X[:, 1], s=0.5, color='red', label='Target')
    plt.scatter(Y[:, 0],  Y[:, 1], s=0.5, color='blue', label='Source')
    plt.legend(loc='upper left', fontsize='x-large')
    plt.draw()

def save_points_ply(xyz, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(path, pcd)
    print("save ply to : ", path)

def compute_point_to_floorplan(base_path, building_name, split="test"):
    json_path = os.path.join(base_path, "json_" + split, building_name + "_floorplan.txt")
    json_points = load_json_to_points(json_path)
    num_points_json = json_points.shape[0]

    # save floorplan ply
    ply_path = os.path.join(base_path, "floorplan_ply_" + split, building_name + "_floorplan.ply")
    save_points_ply(json_points, ply_path)

    points_o3d_path = os.path.join(base_path, "subsampled_points_ply_" + split, building_name + ".ply")
    pcd_points = o3d.io.read_point_cloud(points_o3d_path)
    laz_points = np.array(pcd_points.points)

    if laz_points.shape[0] > 5. * num_points_json:
        choice_idx = np.random.choice(laz_points.shape[0], 5 * num_points_json, replace=False)
        laz_points = laz_points[choice_idx]

    # save original ply
    ply_path = os.path.join(base_path, "points_ply_" + split, building_name + ".ply")
    save_points_ply(laz_points, ply_path)

    mean_json = (json_points.max(0) + json_points.min(0)) / 2.
    mean_laz = (laz_points.max(0) + laz_points.min(0)) / 2.
    move_json_points = json_points - mean_json
    move_laz_points = laz_points - mean_laz

    #import ipdb; ipdb.set_trace()

    '''
    fig = plt.figure()  
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])
    '''
    callback = partial(print_iter)

    reg = Registration(**{'X': move_json_points[:, :2], 'Y': move_laz_points[:, :2]})
    TY, reg_para = reg.register(callback)
    #import ipdb; ipdb.set_trace()
    R, t = reg_para
    #plt.show()

    new_t = - np.dot(mean_laz[:2], R) + t + mean_json[:2]
    transformed_laz = np.dot(laz_points[:, :2], R) + new_t
    laz_points[:, :2] = transformed_laz
    draw_points(json_points, laz_points)

    png_path = os.path.join(base_path, 'matching_png_noscale_' + split, building_name + '.png')
    plt.savefig(png_path, dpi=300)

    #json_dict = {'scale': scale, 'R': R.tolist(), 't': new_t.tolist()}
    json_dict = {'R': R.tolist(), 't': new_t.tolist()}
    point_to_floorplan =  os.path.join(base_path, 'point_to_floorplan_noscale_' + split, building_name + '.json')
    with open(point_to_floorplan, 'w') as f:
        json.dump(json_dict, f)

    # save transformed ply
    ply_path = os.path.join(base_path, "points_transformed_ply_" + split, building_name + "_transformed.ply")
    save_points_ply(laz_points, ply_path)



if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    base_path = "./"


    building_name = '27_Parking_03_F1'

    #compute_point_to_floorplan(base_path, building_name)
    compute_point_to_floorplan(base_path, building_name, split="validation")


    



