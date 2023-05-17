import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6
from load import load_images
from scene_calibration import floor_map, rotation_matrix, get_coordinates_from_point_names, get_extrinsic_matrix_per_camera, get_global_transform_per_camera
import matplotlib.pyplot as plt


def map_test(pattern):

    array_of_floor_points= floor_map(pattern)

    x = [point[0] for point in array_of_floor_points]
    y = [point[1] for point in array_of_floor_points]

    # Create a scatter plot
    plt.scatter(x, y)

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Scatter Plot')
    # Set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Show the plot
    plt.show()


def test(image, image_name, _pattern):
    camera_name = image_name.split('.')[0]
    position = globals()[camera_name]["position"] # 1,2,3 position on the floor
    starting_pair = globals()[camera_name]["start_pair"]

    start_coordinate_system = starting_pair[0]
    #name of the point on the floor , that is our start of coordinate system
    point_name = start_coordinate_system + "_pattern_coordinate_" + str(position) + "_position"
    print(point_name)

    local_to_global_rotation = rotation_matrix(start_coordinate_system)
    print("local_to_global_rotation\n", local_to_global_rotation)

    array_of_floor_points = floor_map(_pattern)

    array_points_name = np.array([
        "A_pattern_coordinate_3_position",
        "B_pattern_coordinate_3_position",
        "C_pattern_coordinate_3_position",
        "D_pattern_coordinate_3_position",
        "A_pattern_coordinate_2_position",
        "B_pattern_coordinate_2_position",
        "C_pattern_coordinate_2_position",
        "D_pattern_coordinate_2_position",
        "A_pattern_coordinate_1_position",
        "B_pattern_coordinate_1_position",
        "C_pattern_coordinate_1_position",
        "D_pattern_coordinate_1_position"
    ])


    index = np.where(array_points_name == point_name)[0]
    if index.size > 0:
        coordinate_system_start = array_of_floor_points[index[0]]
        print("start of the coordinate system - ", point_name,coordinate_system_start,"\n" )

    else:
        print("Value not found in the array.")

def visual_test(image, image_name, pattern):
    camera_name = image_name.split('.')[0]
    position = globals()[camera_name]["position"] # 1,2,3 position on the floor
    starting_pair = globals()[camera_name]["start_pair"]
    letter_start_coordinate_system = starting_pair[0]

    #name of the point on the floor , that is our start of coordinate system
    point_name = letter_start_coordinate_system + "_pattern_coordinate_" + str(position) + "_position"
    print(point_name)
    coordinate_system_start = get_coordinates_from_point_names(point_name, pattern)


    array_of_floor_points= floor_map(_pattern)
    x = [point[0] for point in array_of_floor_points]
    y = [point[1] for point in array_of_floor_points]

    # Create a scatter plot
    plt.scatter(x, y, c = 'b')
    plt.scatter(coordinate_system_start[0], coordinate_system_start[1], c = 'r')

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Scatter Plot')
    # Set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Show the plot
    plt.show()

    extrinsic = get_extrinsic_matrix_per_camera(image,image_name, pattern)
    print("Local extrinsic matrix\n",extrinsic )

    transform_from_local_to_global = get_global_transform_per_camera(point_name, pattern)
    print("Transformation from local to global \n", transform_from_local_to_global)
    rotation_local_to_global  = rotation_matrix(letter_start_coordinate_system)
    print("Rotation from local to global \n", rotation_local_to_global)

    global_extrinsic_matrix = np.matmul(transform_from_local_to_global,np.matmul(rotation_local_to_global, extrinsic))
    print("extrinsic_matrix of camera", camera_name, "-\n", global_extrinsic_matrix, "\n" )



if __name__ == '__main__':
    matplotlib.use('QtCairo')
    images = load_images()
    _pattern = checkered_board

    # map_test(_pattern)

    for key in images:
        image_name = key # for example : cam6.jpg
        image = images[image_name]
        visual_test(image, image_name, _pattern)

    plt.show(block=True)
