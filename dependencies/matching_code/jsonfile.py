import open3d as o3d 
import json
import numpy as np

def transform_json(json_path, R, t, save_path):
    print("Testing IO for load : ", json_path)
    with open(json_path) as f:
        data = json.load(f)

    #print(data.keys())

    header = data['header']
    layer_number = header['layer number']
    structure_number = header['structure number']

    building_points = []
    for layer_id in range(layer_number):
        layer_name = data['layer '+ str(layer_id)]['layer name']
        print("layer id: ", layer_id, "; layer name: ", layer_name)

        pointsets = data['layer '+ str(layer_id)]['points']
        point_coords = []
        total_layer_number_points = 0
        for structure_id in range(structure_number[layer_id]):
            structure_points = data['layer '+ str(layer_id)]['points'][structure_id]
            point_number = structure_points['point number']
            coordinates = np.array(structure_points['coordinates']).reshape(point_number, -1)
            # interpolate points with threshold: th

            new_coordinates = np.dot(coordinates, R) + t
            data['layer '+ str(layer_id)]['points'][structure_id]['coordinates'] = new_coordinates.reshape(-1).tolist()


    data = json.dumps(data, indent = 4)

    with open(save_path, 'w') as f:
        f.write(data)

    return 



def load_json_to_points(json_path, th = 0.5):
    print("Testing IO for load : ", json_path)
    with open(json_path) as f:
        data = json.load(f)

    #print(data.keys())

    header = data['header']
    layer_number = header['layer number']
    structure_number = header['structure number']

    building_points = []
    for layer_id in range(layer_number):
        layer_data = data['layer '+ str(layer_id)]
        layer_name = layer_data['layer name']
        print("layer id: ", layer_id, "; layer name: ", layer_name)

        pointsets = layer_data['points']
        point_coords = []
        total_layer_number_points = 0
        for structure_id in range(structure_number[layer_id]):
            structure_points = pointsets[structure_id]
            point_number = structure_points['point number']
            coordinates = np.array(structure_points['coordinates']).reshape(point_number, -1)
            # interpolate points with threshold: th
            new_coordinates = [coordinates[0]]
            for seg_id in range(point_number - 1):
                start_x, start_y = coordinates[seg_id][0], coordinates[seg_id][1]
                end_x, end_y = coordinates[seg_id + 1][0], coordinates[seg_id + 1][1]
                dist = np.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)
                number_points_add = int(np.floor(dist / th))
                if number_points_add == 0:
                    new_coordinates.append(np.array([end_x, end_y]))
                    continue

                dx, dy = (end_x - start_x) / number_points_add, (end_y - start_y) / number_points_add
                for point_id in range(number_points_add):
                    new_coordinates.append(np.array([start_x + dx * (point_id + 1), start_y + dy * (point_id + 1)]))

            point_coords += new_coordinates
            total_layer_number_points += len(new_coordinates)

        #import ipdb; ipdb.set_trace()
        point_coords = np.stack(point_coords, axis = 0)
        Z = np.zeros((total_layer_number_points, 1))
        point_coords_3d = np.concatenate([point_coords, Z], axis = -1)
        building_points.append(point_coords_3d)


    building_points = np.concatenate(building_points, axis = 0)
    #pcd.points = o3d.utility.Vector3dVector(building_points)
    #o3d.visualization.draw_geometries([pcd])
    return building_points

def load_transformation(json_path):
    print("load transformation : ", json_path)
    with open(json_path) as f:
        data = json.load(f)

    scale = data['scale']
    R = np.array(data['R'])
    t = np.array(data['t'])
    return scale, R, t

def load_transformation_affine(json_path):
    print("load transformation : ", json_path)
    with open(json_path) as f:
        data = json.load(f)

    R = np.array(data['R'])
    t = np.array(data['t'])
    return R, t



if __name__=="__main__":

    json_path = "./json/01_OfficeLab_01_F1_floorplan.txt"
    save_path = "./transformed_json/01_OfficeLab_01_F1_floorplan.txt"

    R = np.array([[0.8954028730992107, -0.4452568863540224], [0.44525688635402233, 0.895402873099211]])
    #R = np.array([[1.0, 0], [0, 1.0]])
    t = np.array([-824.6887914928739 + 6.54 + 15.4, -274.03160108145187 + 2.42 + 12])


    transform_json(json_path, R, t, save_path)


   
    




