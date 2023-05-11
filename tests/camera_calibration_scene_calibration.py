import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6
from load import load_images
from camera_calibration.vanishing_points import principal_point_coordinates

import matplotlib.pyplot as plt




def floor_map(_pattern):
    x_pattern_size, y_pattern_size = _pattern["size"]
    x_blank = _pattern["x_blank"]
    y_blank = _pattern["y_blank"]
    distance_between_patterns = 3000  # 3 m = 3000 mm
    pattern_actual_x = x_pattern_size - 2 * x_blank
    pattern_actual_y = y_pattern_size - 2 * y_blank
    D_edge_coordinate_3_position = np.array([2000, 2000, 0])  # 2m , 2m ,0 x, y, z
    A_edge_coordinate_3_position = D_edge_coordinate_3_position - np.array([x_pattern_size, y_pattern_size, 0])
    A_pattern_coordinate_3_position = A_edge_coordinate_3_position + np.array([x_blank, y_blank, 0])
    B_pattern_coordinate_3_position = A_pattern_coordinate_3_position + np.array([0, pattern_actual_y, 0])
    C_pattern_coordinate_3_position = B_pattern_coordinate_3_position + np.array([pattern_actual_x, 0, 0])
    D_pattern_coordinate_3_position = A_pattern_coordinate_3_position + np.array([pattern_actual_x, 0, 0])

    A_edge_coordinate_2_position = A_edge_coordinate_3_position + np.array(
        [0, y_pattern_size + distance_between_patterns, 0])
    A_pattern_coordinate_2_position = A_edge_coordinate_2_position + np.array([x_blank, y_blank, 0])
    B_pattern_coordinate_2_position = A_pattern_coordinate_2_position + np.array([0, pattern_actual_y, 0])
    C_pattern_coordinate_2_position = B_pattern_coordinate_2_position + np.array([pattern_actual_x, 0, 0])
    D_pattern_coordinate_2_position = A_pattern_coordinate_2_position + np.array([pattern_actual_x, 0, 0])

    A_edge_coordinate_1_position = A_edge_coordinate_2_position + np.array(
        [0, y_pattern_size + distance_between_patterns, 0])
    A_pattern_coordinate_1_position = A_edge_coordinate_1_position + np.array([x_blank, y_blank, 0])
    B_pattern_coordinate_1_position = A_pattern_coordinate_1_position + np.array([0, pattern_actual_y, 0])
    C_pattern_coordinate_1_position = B_pattern_coordinate_1_position + np.array([pattern_actual_x, 0, 0])
    D_pattern_coordinate_1_position = A_pattern_coordinate_1_position + np.array([pattern_actual_x, 0, 0])

    array_of_points = np.array([A_pattern_coordinate_3_position,
                                B_pattern_coordinate_3_position,
                                C_pattern_coordinate_3_position,
                                D_pattern_coordinate_3_position,

                                A_pattern_coordinate_2_position,
                                B_pattern_coordinate_2_position,
                                C_pattern_coordinate_2_position,
                                D_pattern_coordinate_2_position,

                                A_pattern_coordinate_1_position,
                                B_pattern_coordinate_1_position,
                                C_pattern_coordinate_1_position,
                                D_pattern_coordinate_1_position])
    return array_of_points

def map_test(pattern):

    array_of_floor_points= floor_map(pattern)

    for i in array_of_floor_points:
        print(i[0], i[1], "\n")
    # Extract coordinates from the points
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


def visual_test(image, image_name, _pattern):
    camera_name = image_name.split('.')[0]
    position = globals()[camera_name]["position"] # 1,2,3 position on the floor
    starting_pair = globals()[camera_name]["start_pair"]

    start_coordinate_system = starting_pair[0]
    #name of the point on the floor , that is our start of coordinate system
    point_name = start_coordinate_system + "_pattern_coordinate_" + str(position) + "_position"
    print(point_name)


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





if __name__ == '__main__':
    matplotlib.use('QtCairo')
    images = load_images()
    _pattern = checkered_board

    # map_test(_pattern)

    for key in images:
        image_name = key # for example : cam6.jpg
        image = images[image_name]
        visual_test(image, image_name, _pattern)
        # x, y, z = get_distance_to_calibration_pattern(image,image_name, _pattern)
        # assert ((vt_oa_ro - np.linalg.norm([x, y, z])) < 1e-12)
        #
        # print(f"Distance to the wall: {y/1000:0.2f}m, "
        #       f"distance from the focal axis: {x/1000:0.2f}m, "
        #       f"camera installation height: {z/1000:0.2f}m")
        # print("\n")

    plt.show(block=True)




    plt.show(block=True)
