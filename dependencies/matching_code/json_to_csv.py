import os 

import numpy as np 
import json
from jsonfile import load_transformation_affine as load_transformation
import csv

if __name__ == "__main__":
    #import ipdb; ipdb.set_trace()

    base_path = "./"

    split = "test"
    
    file_list = os.path.join(base_path, 'filelist_'+split+'.txt')

    with open(file_list, 'r') as f:
        building_names = f.readlines()

    transformation = {}

    for building_idx, building_name in enumerate(building_names):

        building_name = building_name.rstrip().replace('.json', '')

        print('===============================================================')
        print('============working on : ', building_idx, " ------ ", building_name)
        print('===============================================================')

        transformation_path = os.path.join(base_path, "point_to_floorplan_noscale_" + split, building_name + ".json")
        R, t = load_transformation(transformation_path)
        transformation[building_name] = {'R': R, 't': t}


    with open('transformation_rigid_'+split+'.csv', mode='w') as csv_file:
        fieldnames = ['scene_name', 'rotation (1,1)', 'rotation (1,2)', 'rotation (2,1)', 'rotation (2,2)', 'translation x', 'translation y']

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        #import ipdb; ipdb.set_trace()
        for key, value in transformation.items():

            print(key)
            tmp = {'scene_name': key,
                'rotation (1,1)' : value['R'][0, 0], 
                'rotation (1,2)' : value['R'][0, 1], 
                'rotation (2,1)' : value['R'][1, 0], 
                'rotation (2,2)' : value['R'][1, 1], 
                'translation x' : value['t'][0], 
                'translation y' : value['t'][1]}
            writer.writerow(tmp)

    print('done!!!')


