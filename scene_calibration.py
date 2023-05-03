import numpy as np

from camera_calibration.vanishing_points import get_distance_to_calibration_pattern, get_rotation
from load import load_images
from callibration_patterns import checkered_board


def get_extrinsic_matrix_per_camera(img, pattern):
    rotation_matrix = get_rotation(img, pattern)
    translation_vector = np.array(get_distance_to_calibration_pattern(img, pattern)).reshape((3,))
    translation_vector = translation_vector/1000 #mm into m
    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[:3,:3]= rotation_matrix
    extrinsic_matrix[:3,3] = translation_vector
    return extrinsic_matrix

def get_global_transform_per_camera(camera_name, pattern):
    camera_number = int(camera_name.split(".")[0][-1])
    x_pattern_size, y_pattern_size = pattern["size"]
    x_blank = pattern["x_blank"]
    y_blank = pattern["y_blank"]
    distance_between_patterns = 3000 # 3 m = 3000 mm

    translation_matrix = np.identity(4)
    D_edge_coordinate_3_position = np.array([2000, 2000, 0]) # 2m , 2m ,0
    A_edge_coordinate_3_position = D_edge_coordinate_3_position - np.array([x_pattern_size, y_pattern_size, 0])
    A_pattern_coordinate_3_position = A_edge_coordinate_3_position + np.array([x_blank, y_blank, 0])

    A_edge_coordinate_2_position = A_edge_coordinate_3_position + np.array([0,y_pattern_size+distance_between_patterns, 0])
    A_pattern_coordinate_2_position = A_edge_coordinate_2_position + np.array([x_blank, y_blank, 0])

    A_edge_coordinate_1_position = A_edge_coordinate_2_position + np.array([0,y_pattern_size+distance_between_patterns, 0])
    A_pattern_coordinate_1_position = A_edge_coordinate_1_position + np.array([x_blank, y_blank, 0])

    #3rd position of the pattern
    if (camera_number == 1) or (camera_number == 2):
        translation_matrix[:3,3] = - A_pattern_coordinate_3_position

    # 2nd position of the pattern
    if (camera_number == 3) or (camera_number == 6):
        translation_matrix[:3,3] = - A_pattern_coordinate_2_position

    # 1st position of the pattern
    if (camera_number == 4) or (camera_number == 5):
        translation_matrix[:3,3] = - A_pattern_coordinate_1_position

    return translation_matrix

def get_global_extrisic_matrix(img, image_name,pattern):
    extrinsic = get_extrinsic_matrix_per_camera(img, pattern)
    transform_from_local_to_global = get_global_transform_per_camera(image_name, pattern)
    global_extrinsic_matrix = np.matmul(transform_from_local_to_global, extrinsic)
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