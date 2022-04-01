import numpy as np
import open3d as o3d 
import os
import copy
import matplotlib.pyplot as plt
import json
import glob
import csv
import matplotlib
from jsonfile import load_json_to_points



def draw_points(X, Y):
    plt.cla()
    plt.scatter(X[:, 0],  X[:, 1], s=0.5, color='red', label='floorplan')
    plt.scatter(Y[:, 0],  Y[:, 1], s=0.5, color='blue', label='pointcloud')
    plt.legend(loc='upper left', fontsize='x-large')
    plt.draw()

def draw_save_points(X, Y, path):
    plt.cla()
    plt.scatter(X[:, 0],  X[:, 1], s=0.5, color='red', label='floorplan')
    plt.scatter(Y[:, 0],  Y[:, 1], s=0.5, color='blue', label='pointcloud')
    plt.legend(loc='upper left', fontsize='x-large')
    plt.axis('equal')
    plt.draw()
    plt.savefig(path, dpi=300)

def save_buildings_transformed_json(base_path, building_name, split):
    json_path = os.path.join(base_path, "transformed_json_" + split, building_name + "_floorplan.txt")
    json_points = load_json_to_points(json_path)

    points_o3d_path = os.path.join(base_path, "points_ply_" + split, building_name + ".ply")
    pcd_points = o3d.io.read_point_cloud(points_o3d_path)
    laz_points = np.array(pcd_points.points)

    png_path = os.path.join(base_path, 'final_transformed_json_png_'+split, building_name + '.png')
    draw_save_points(json_points, laz_points, png_path)

    print("done!!!")


if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    base_path = "./"

    #building_name = "01_OfficeLab_01_F1"
    split='test'

    rigid = {}
    with open("./transformation_rigid_"+ split +"_final.csv", 'r') as data:
        for line in csv.DictReader(data):
            scene_name = line['scene_name']
            R = np.zeros((2, 2))
            R[0, 0] = line['rotation (1,1)']
            R[0, 1] = line['rotation (1,2)']
            R[1, 0] = line['rotation (2,1)']
            R[1, 1] = line['rotation (2,2)']
            t = np.zeros(2)
            t[0] = line['translation x']
            t[1] = line['translation y']
            rigid[scene_name] = {'R': np.array(R), 't': np.array(t)}

    for key in rigid:

        save_buildings_transformed_json(base_path, key, split=split)


    



