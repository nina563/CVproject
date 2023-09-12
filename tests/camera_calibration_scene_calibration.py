import numpy as np
import matplotlib
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6
from load import load_images
from scene_calibration import floor_map, rotation_matrix, get_coordinates_from_point_names, get_extrinsic_matrix_per_camera, get_local_to_global_transform, get_global_extrisic_matrix
import matplotlib.pyplot as plt
from camera_calibration.vanishing_points import get_distance_to_calibration_pattern, get_intrinsic_matrix


def map_test(pattern):

    array_of_floor_points= floor_map(pattern)

    x = [point[0] for point in array_of_floor_points]
    y = [point[1] for point in array_of_floor_points]

    names = ["A",
        "B","C", "D", "A",
        "B","C", "D", "A",
        "B","C", "D"]

    # Plot the points
    plt.scatter(x, y)

    # Plot the names of the points
    for i in range(len(x)):
        plt.text(x[i], y[i], names[i], fontsize=12, ha='center', va='bottom')

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
    print("letter_start_coordinate_system", letter_start_coordinate_system)
    coordinate_system_start = get_coordinates_from_point_names(point_name, pattern)


    array_of_floor_points= floor_map(_pattern)
    x = [point[0] for point in array_of_floor_points]
    y = [point[1] for point in array_of_floor_points]

    plt.scatter(x, y, c = 'b')
    plt.scatter(coordinate_system_start[0], coordinate_system_start[1], c = 'r')

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Scatter Plot')
    # Set aspect ratio
    plt.gca().grid("on")
    plt.gca().set_aspect('equal', adjustable='box')

    # Show the plot
    plt.show()


def test_mapping(images, pattern):

    array_of_floor_points = floor_map(pattern)
    position_3 = array_of_floor_points[:4]
    position_2 = array_of_floor_points[4:8]
    position_1 = array_of_floor_points[8:12]

    x_pattern_size, y_pattern_size = _pattern["size"]
    x_blank = _pattern["x_blank"]
    y_blank = _pattern["y_blank"]
    pattern_actual_x = x_pattern_size - 2 * x_blank
    pattern_actual_y = y_pattern_size - 2 * y_blank

    print("cam1")
    A = np.array([0, 0, 0, 1])
    B = np.array([0,pattern_actual_x, 0,1])
    C = np.array([pattern_actual_y,pattern_actual_x, 0,1])
    D = np.array([pattern_actual_y,0, 0,1])
    rotation_local_to_global, transform_from_local_to_global = get_local_to_global_transform("cam1.jpg", pattern)
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, B))[:3] == position_3[0])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, C))[:3] == position_3[1])
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, D))[:3] == position_3[2])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, A))[:3] == position_3[3])

    print("cam2")
    B = np.array([0,pattern_actual_y, 0,1])
    C = np.array([pattern_actual_x,pattern_actual_y, 0,1])
    D = np.array([pattern_actual_x,0, 0,1])
    rotation_local_to_global, transform_from_local_to_global = get_local_to_global_transform("cam2.jpg", pattern)
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, A))[:3] == position_3[0])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, B))[:3] == position_3[1])
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, C))[:3] == position_3[2])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global,D))[:3] == position_3[3])

    print("cam3")
    B = np.array([0,pattern_actual_y, 0,1])
    C = np.array([pattern_actual_x,pattern_actual_y, 0,1])
    D = np.array([pattern_actual_x,0, 0,1])
    rotation_local_to_global, transform_from_local_to_global = get_local_to_global_transform("cam3.jpg", pattern)
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, A))[:3] == position_2[0])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, B))[:3] == position_2[1])
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, C))[:3] == position_2[2])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global,D))[:3] == position_2[3])


    print("cam4")
    B = np.array([0,pattern_actual_x, 0,1])
    C = np.array([pattern_actual_y,pattern_actual_x, 0,1])
    D = np.array([pattern_actual_y,0, 0,1])
    rotation_local_to_global, transform_from_local_to_global = get_local_to_global_transform("cam4.jpg", pattern)
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, D))[:3] == position_1[0])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, A))[:3] == position_1[1])
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, B))[:3] == position_1[2])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global,C))[:3] == position_1[3])

    print("cam5")
    B = np.array([0,pattern_actual_y, 0,1])
    C = np.array([pattern_actual_x,pattern_actual_y, 0,1])
    D = np.array([pattern_actual_x,0, 0,1])
    rotation_local_to_global, transform_from_local_to_global = get_local_to_global_transform("cam5.jpg", pattern)
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, C))[:3] == position_1[0])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, D))[:3] == position_1[1])
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, A))[:3] == position_1[2])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global,B))[:3] == position_1[3])

    print("cam6")
    A = np.array([0, 0, 0, 1])
    B = np.array([0,pattern_actual_x, 0,1])
    C = np.array([pattern_actual_y,pattern_actual_x, 0,1])
    D = np.array([pattern_actual_y,0, 0,1])
    rotation_local_to_global, transform_from_local_to_global = get_local_to_global_transform("cam6.jpg", pattern)
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, B))[:3] == position_2[0])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, C))[:3] == position_2[1])
    print( np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, D))[:3] == position_2[2])
    print(np.matmul(transform_from_local_to_global, np.matmul(rotation_local_to_global, A))[:3] == position_2[3])

def save_matrix(images, _pattern):

    with open('extrinsic_intrinsic', 'a', encoding='utf-8') as file_re:
        for key in images:
            image_name = key  # for example : cam6.jpg
            image = images[image_name]
            global_extrisic_matrix = get_global_extrisic_matrix(image, image_name, _pattern)
            intrinsic_matrix = get_intrinsic_matrix(image, image_name, _pattern)
            string = "Image: {}\nIntrinsic matrix:\n {} \nGlobal extrinsic matrix:\n {} \n".format(image_name,intrinsic_matrix,
                                                                                                        global_extrisic_matrix)
            file_re.write(string + '\n')


if __name__ == '__main__':
    matplotlib.use('QtCairo')
    images = load_images()
    _pattern = checkered_board
    # test_mapping(images, _pattern)
    # save_matrix(images, _pattern)
    # map_test(_pattern)
    #
    for key in images:
        image_name = key # for example : cam6.jpg
        image = images[image_name]
        visual_test(image, image_name, _pattern)

    plt.show(block=True)
