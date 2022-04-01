import os
import open3d as o3d 
import numpy as np 
from laspy.file import File

'''
if __name__ == "__main__":

    import ipdb; ipdb.set_trace()

    
    inFile = File('./LAS/01_OfficeLab_01_F1_s0p01m.laz', mode='r')
    scale_x, scale_y, scale_z = inFile.header.scale
    offset_x, offset_y, offset_z = inFile.header.offset 

    #load coordinates
    X, Y, Z = inFile.X * scale_x + offset_x, inFile.Y * scale_y + offset_y, inFile.Z * scale_z + offset_z
    coord = np.stack([X, Y, Z], axis = -1).astype(np.float32)

    num_choice = 100000
    choice_idx = np.random.choice(coord.shape[0], num_choice, replace= False)
    coord = coord[choice_idx]

    mean_Z = coord.mean(0)[-1]
    z_th = 0.5 
    points = coord[coord[:, 2] > (mean_Z - z_th)]
    points = points[points[:, 2] < (mean_Z + z_th)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
'''
    
def load_laz_to_points(laz_path):
    print("load laz file : ", laz_path)
    inFile = File(laz_path, mode='r')
    scale_x, scale_y, scale_z = inFile.header.scale
    offset_x, offset_y, offset_z = inFile.header.offset 

    #load coordinates
    X, Y, Z = inFile.X * scale_x + offset_x, inFile.Y * scale_y + offset_y, inFile.Z * scale_z + offset_z
    coord = np.stack([X, Y, Z], axis = -1).astype(np.float32)

    #import ipdb; ipdb.set_trace()

    num_choice = 100000
    choice_idx = np.random.choice(coord.shape[0], num_choice, replace= False)
    coord = coord[choice_idx]

    mean_Z = coord.mean(0)[-1]
    z_th = 0.2
    points = coord[coord[:, 2] > (mean_Z - z_th)]
    points = points[points[:, 2] < (mean_Z + z_th)]
    points[:, 2] = 0.

    return points


if __name__ == "__main__":
    #import ipdb; ipdb.set_trace()

    base_path = "./"
    
    file_list = os.path.join(base_path, 'file_list_test.txt')

    with open(file_list, 'r') as f:
        building_names = f.readlines()
    
    #building_names = ['01_OfficeLab_01']

    dispaly = False

    save_path = "subsampled_points_ply_test"

    for building_idx, building_name in enumerate(building_names):

        building_name = building_name.rstrip().replace('_floorplan.txt', '')
        try:
            laz_path = os.path.join(base_path, 'LAS_test', building_name + '_s0p01m.laz')

            print('===============================================================')
            print('============working on : ', building_idx, " ------ ", building_name)
            print('===============================================================')

            laz_points = load_laz_to_points(laz_path)

        except:
            laz_path = os.path.join(base_path, 'LAS_test', building_name + '_s0p01m.LAZ')

            print('===============================================================')
            print('============working on : ', building_idx, " ------ ", building_name)
            print('===============================================================')

            laz_points = load_laz_to_points(laz_path)

        save_file = os.path.join(base_path, save_path, building_name + '.ply')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(laz_points)
        o3d.io.write_point_cloud(save_file, pcd)
    
        if dispaly:
            o3d.visualization.draw_geometries([pcd])

'''

if __name__ == "__main__":

    import ipdb; ipdb.set_trace()

    building_name = '27_Parking_03_F2'
    inFile = File('./LAS_validation/'+ building_name +'_s0p01m.laz', mode='r')
    scale_x, scale_y, scale_z = inFile.header.scale
    offset_x, offset_y, offset_z = inFile.header.offset 

    pcd = o3d.geometry.PointCloud()

    #load coordinates
    X, Y, Z = inFile.X * scale_x + offset_x, inFile.Y * scale_y + offset_y, inFile.Z * scale_z + offset_z
    coord = np.stack([X, Y, Z], axis = -1).astype(np.float32)

    num_choice = 100000
    choice_idx = np.random.choice(coord.shape[0], num_choice, replace= False)
    coord = coord[choice_idx]

    mean_Z = 4.#coord.min(0)[-1]
    z_th = 0.2
    points = coord[coord[:, 2] > (mean_Z - z_th)]
    points = points[points[:, 2] < (mean_Z + z_th)]
    #points = coord

    
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


    save_file = os.path.join('./subsampled_points_ply_validation', building_name + '.ply')
    o3d.io.write_point_cloud(save_file, pcd)
'''