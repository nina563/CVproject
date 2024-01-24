from scene_calibration import floor_map, get_global_transform_per_camera, rotation_matrix,get_rotation, get_global_extrisic_matrix
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6

from camera_calibration.vanishing_points import get_distance_to_calibration_pattern
import numpy as np
from load import load_images
from math import asin, pi, atan2, cos, copysign
import cv2
import pickle
import matplotlib.pyplot as plt
import json
from pathlib import Path

#returns
def get_global_camera_position(test_image,image_name, pattern):
    camera_name = image_name.split('.')[0]
    position = globals()[camera_name]["position"]  # 1,2,3 position on the floor
    starting_pair = globals()[camera_name]["start_pair"]
    letter_start_coordinate_system = starting_pair[0]

    # name of the point on the floor , that is our start of coordinate system
    point_name = letter_start_coordinate_system + "_pattern_coordinate_" + str(position) + "_position"

    transform_from_local_to_global = get_global_transform_per_camera(point_name, pattern)
    rotation_local_to_global = rotation_matrix(letter_start_coordinate_system)
    local_camera_pos = get_distance_to_calibration_pattern(test_image,image_name, pattern)# in mm
    reshaped_array = np.append(local_camera_pos, 1)
    # Reshape the modified array to shape (4, 1)
    reshaped_local_camera_pos = np.reshape(reshaped_array, (4, 1))

    global_local_camera = np.matmul(transform_from_local_to_global,np.matmul(rotation_local_to_global, reshaped_local_camera_pos))
    global_local_camera= global_local_camera/global_local_camera[-1]
    return np.squeeze(global_local_camera[:3])

def get_global_rotation(test_image,image_name, pattern):
    camera_name = image_name.split('.')[0]
    position = globals()[camera_name]["position"]  # 1,2,3 position on the floor
    starting_pair = globals()[camera_name]["start_pair"]
    letter_start_coordinate_system = starting_pair[0]

    # name of the point on the floor , that is our start of coordinate system
    point_name = letter_start_coordinate_system + "_pattern_coordinate_" + str(position) + "_position"

    transform_from_local_to_global = get_global_transform_per_camera(point_name, pattern)
    rotation_local_to_global = rotation_matrix(letter_start_coordinate_system)
    local_rotationmatrix = get_rotation(test_image, image_name, pattern)
    angles = rotation_matrix_to_euler_angles(local_rotationmatrix)

    #reshapping 3 by 3 array into 4 by 4
    reshapped_local_rotationmatrix = np.identity(4)
    reshapped_local_rotationmatrix[:3,:3] = local_rotationmatrix
    k_vector = np.array([0,0,1])
    inverse =np.linalg.inv(local_rotationmatrix)
    local_3d_cam_dir = np.matmul(inverse, k_vector)
    local_3d_cam_dir = np.append(local_3d_cam_dir,1)
    global_cam_vector_k = np.matmul(rotation_local_to_global, local_3d_cam_dir)
    global_cam_vector_k = global_cam_vector_k[:3]

    x,y,z = global_cam_vector_k

    # x_rot = np.arctan2(np.abs(y),z)*180/pi-90
    x_rot =- np.arctan2(np.abs(y), z) * 180 / pi
    y_rot = 0
    z_rot = -np.arctan2(x, y) * 180 / pi

    rotation = np.array([x_rot, y_rot, z_rot ])
    return rotation


def calculate_average(lst):
    total_sum = sum(lst)
    list_length = len(lst)
    if list_length == 0:
        return 0
    else:
        average = total_sum / list_length
        return average

def get_cam_parameters_dict(pattern):
    # Open the video file
    filelist = list(Path('.').glob('**/*.mp4'))
    camera_position_dict = {}
    camera_rotation_dict = {}
    extrinsic_matrix_dict = {}
    for i in filelist:
        filename = i.name
        camera_name = filename.split('.')[0]  # Split the filename by '.' and get the first element
        vid = cv2.VideoCapture(str(i))

        frame_rate = vid.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
        skip_duration = int(frame_rate)  # Set the skip duration in seconds (1 second in this case)

        # Calculate the number of frames to skip
        frames_to_skip = int(skip_duration * frame_rate)

        index = 0
        # Skip frames to reach the desired start point
        for _ in range(frames_to_skip):
            vid.read()
        while (index < 20):
            # Extract images
            ret, frame = vid.read()
            if not ret:
                break

            image_name = filename
            camera_position = get_global_camera_position(frame, image_name, pattern)
            camera_rotation = get_global_rotation(frame,image_name, pattern)
            extrinsic_matrix = get_global_extrisic_matrix(frame, image_name, pattern)
            if camera_name in extrinsic_matrix_dict:
                # Camera already exists in the dictionary
                extrinsic_matrix_dict[camera_name].append(extrinsic_matrix)
            else:
                # Camera does not exist in the dictionary
                extrinsic_matrix_dict[camera_name] = [extrinsic_matrix]

            # next frame
            index += 1

    file_path_pos = "nina_data/camera_pos_dict.pkl"
    file_path_rot = "nina_data/camera_rotation_dict.pkl"
    file_path_ext = "nina_data/extrinsic_matrix_dict.pkl"
    with open(file_path_ext, 'wb') as file:
        # Serialize and save the dictionary using pickle
        pickle.dump(extrinsic_matrix_dict, file)

def get_av_cam_parameters():

    camera_av_parameters = {}

    with open('nina_data/camera_pos_dict.pkl', 'rb') as file:
        pos_dictionary= pickle.load(file)
        for camera_name, camera_positions in pos_dictionary.items():
            # Extract x and y coordinates from the list of tuples
            x_values = [coord[0] for coord in camera_positions]
            y_values = [coord[1] for coord in camera_positions]
            z_values = [coord[2] for coord in camera_positions]

            av_coordinates=np.array([calculate_average(x_values), calculate_average(y_values), calculate_average(z_values)])

            std_coordinates = np.array([np.std(x_values),np.std(y_values),np.std(z_values)])

            if camera_name not in camera_av_parameters:
                camera_av_parameters[camera_name] = {}
            camera_av_parameters[camera_name]['av_positions'] = av_coordinates.tolist()
            camera_av_parameters[camera_name]['std_positions'] = std_coordinates.tolist()

    with open('nina_data/camera_rotation_dict.pkl', 'rb') as file:
        pos_dictionary= pickle.load(file)
        for camera_name, camera_rotation in pos_dictionary.items():
            # Extract x and y coordinates from the list of tuples
            x = [angle[0] for angle in camera_rotation]
            y = [angle[1] for angle in camera_rotation]
            z = [angle[2] for angle in camera_rotation]
            av_angles=np.array([calculate_average(x), calculate_average(y), calculate_average(z)])
            std_angles = np.array([np.std(x), np.std(y), np.std(z)])

            if camera_name not in camera_av_parameters:
                camera_av_parameters[camera_name] = {}
            camera_av_parameters[camera_name]['av_angles'] = av_angles.tolist()
            camera_av_parameters[camera_name]['std_angles'] = std_angles.tolist()


    with open('nina_data/extrinsic_matrix_dict.pkl', 'rb') as file:
        extrinsic_matrix_dict= pickle.load(file)

        for camera_name, extrinsic_matrix in extrinsic_matrix_dict.items():
            average_matrix = np.mean(extrinsic_matrix, axis=0)[:3,:3]

            if camera_name not in camera_av_parameters:
                camera_av_parameters[camera_name] = {}
            camera_av_parameters[camera_name]['extrinsic_matrix'] = average_matrix.tolist()

    # Sort the dictionary by camera names
    camera_av_parameters = dict(sorted(camera_av_parameters.items()))

    file_path_pos = "nina_data/camera_av_parameters.json"
    with open(file_path_pos, 'w') as file:
        json.dump(camera_av_parameters, file, indent=4)

    print("Data saved to camera_av_parameters.json")


def plot_dict(_pattern):
    array_of_points = floor_map(_pattern)

    x = [point[0] for point in array_of_points]
    y = [point[1] for point in array_of_points]

    names = ["A",
        "B","C", "D", "A",
        "B","C", "D", "A",
        "B","C", "D"]

    # Plot the points
    plt.scatter(x, y)

    # Plot the names of the points
    for i in range(len(x)):
        plt.text(x[i], y[i], names[i], fontsize=8, ha='center', va='bottom')


    with open('nina_data/camera_pos_dict.pkl', 'rb') as file:
        dictionary= pickle.load(file)
        for key, coordinates in dictionary.items():
            # Extract x and y coordinates from the list of tuples
            x_values = [coord[0] for coord in coordinates]
            y_values = [coord[1] for coord in coordinates]
            z_values = [coord[2] for coord in coordinates]

            # Plot the points for each camera
            plt.scatter(x_values, y_values, label=key)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Show the plot
    plt.gca().grid("on")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('plot.jpg', format='jpg')
    plt.show()

def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = np.sqrt(rotation_matrix[2, 1] * rotation_matrix[2, 1] + rotation_matrix[2, 2] * rotation_matrix[2, 2])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.array([x, y, z])*180/pi



if __name__ == '__main__':
    images = load_images()
    _pattern = checkered_board












