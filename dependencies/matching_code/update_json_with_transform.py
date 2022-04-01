import os
import csv
import json
import numpy as np
from jsonfile import transform_json

if __name__ == "__main__":
    transformation_path = "./transformation_rigid_test_final.csv"

    rigid = {}
    with open(transformation_path, 'r') as data:
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
        print("----- scene_name : ", key, " -----")

        R, t = rigid[key]['R'], rigid[key]['t']

        json_path = os.path.join('./', 'json_test', key + '_floorplan.txt')

        save_path = os.path.join('./', 'transformed_json_test', key + '_floorplan.txt')

        transform_json(json_path, R, t, save_path)






