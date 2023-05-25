import numpy as np

from camera_calibration.vanishing_points import get_distance_to_calibration_pattern, get_rotation
from load import load_images
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6


def floor_map(_pattern):
    x_pattern_size, y_pattern_size = _pattern["size"] 
    x_blank = _pattern["x_blank"]
    y_blank = _pattern["y_blank"]
    distance_between_patterns = 3000  # 3 m = 3000 mm
    pattern_actual_x = x_pattern_size - 2 * x_blank
    pattern_actual_y = y_pattern_size - 2 * y_blank
    D_edge_coordinate_3_position = np.array([2000, 2000, 0])  # 2m , 2m ,0 x, y, z
    A_edge_coordinate_3_position = D_edge_coordinate_3_position - np.array([x_pattern_size, 0, 0])
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

def get_coordinates_from_point_names(point_name,_pattern):
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
    array_of_floor_points = floor_map(_pattern)
    index = np.where(array_points_name == point_name)[0]
    if index.size > 0:
        coordinate_system_start = array_of_floor_points[index[0]]
        # print("start of the coordinate system - ", point_name,coordinate_system_start )
    else:
        print("Value not found in the array.")

    return coordinate_system_start


def get_extrinsic_matrix_per_camera(img,image_name, pattern):
    rotation_matrix = get_rotation(img, image_name,pattern)
    translation_vector = np.array(get_distance_to_calibration_pattern(img,image_name, pattern)).reshape((3,))
    translation_vector = translation_vector
    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[:3,:3]= rotation_matrix
    extrinsic_matrix[:3,3] = translation_vector
    return extrinsic_matrix

def local_rotation(angle):
    tol = 1e-8
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)

    # Set values close to zero as zero
    if np.abs(cos_val) < tol:
        cos_val = 0.0
    if np.abs(sin_val) < tol:
        sin_val = 0.0

    rotation = np.array([[cos_val, -sin_val, 0],
                         [sin_val, cos_val , 0],
                         [0              , 0               , 1]])
    return rotation


def rotation_matrix(start_coordinate_system):
    rotation_matrix = np.identity(4)
    if start_coordinate_system == "A":
        rotation_matrix[:3,:3] = local_rotation(0)
    elif start_coordinate_system == "B":
        rotation_matrix[:3, :3] = local_rotation(np.pi/2)
    elif start_coordinate_system == "C":
        rotation_matrix[:3,:3] = local_rotation(np.pi)
    else:
        rotation_matrix[:3,:3] = local_rotation(3*np.pi/2)

    return rotation_matrix



def get_global_transform_per_camera(point_name, pattern):
    translation_matrix = np.identity(4)
    coordinate_system_start = get_coordinates_from_point_names(point_name, pattern)
    translation_matrix[:3, 3] = - coordinate_system_start

    return translation_matrix

def get_global_extrisic_matrix(img, image_name,pattern):
    extrinsic = get_extrinsic_matrix_per_camera(img,image_name, pattern)

    camera_name = image_name.split('.')[0]
    position = globals()[camera_name]["position"] # 1,2,3 position on the floor
    starting_pair = globals()[camera_name]["start_pair"]
    letter_start_coordinate_system = starting_pair[0]

    #name of the point on the floor , that is our start of coordinate system
    point_name = letter_start_coordinate_system + "_pattern_coordinate_" + str(position) + "_position"
    # print("start of the coordinate system", point_name)


    transform_from_local_to_global = get_global_transform_per_camera(point_name, pattern)
    rotation_local_to_global  = rotation_matrix(letter_start_coordinate_system)

    global_extrinsic_matrix = np.matmul(transform_from_local_to_global,np.matmul(rotation_local_to_global, extrinsic))
    # print("extrinsic_matrix of camera", camera_name, "-\n", global_extrinsic_matrix, "\n" )
    return global_extrinsic_matrix

def global_extrisic_matrix_for_all():
    images = load_images()
    _pattern = checkered_board
    for key in images:
        image_name = key # for example : cam6.jpg
        image = images[image_name]
        get_global_extrisic_matrix(image, image_name,_pattern )


if __name__ == '__main__':
    images = load_images()
    _pattern = checkered_board

    global_extrinsic_matrix = get_global_extrisic_matrix(images["cam2.jpg"],"cam2.jpg", _pattern)
    global_extrisic_matrix_for_all()
