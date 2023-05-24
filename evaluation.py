from scene_calibration import get_global_transform_per_camera, rotation_matrix
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6
from camera_calibration.vanishing_points import get_distance_to_calibration_pattern
import numpy as np
from load import load_images
import cv2
from pathlib import Path
import os
import pickle
import json


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


def process_video(pattern):
    # Open the video file
    filelist = list(Path('.').glob('**/*.mp4'))
    camera_dict = {}
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

            # Append coordinates for the camera multiple times
            if camera_name in camera_dict:
                # Camera already exists in the dictionary
                camera_dict[camera_name].append(camera_position)
            else:
                # Camera does not exist in the dictionary
                camera_dict[camera_name] = [camera_position]

            # next frame
            print(camera_name, index)
            index += 2

    file_path = "camera_pos_dict.pkl"

    # Open the file in write mode
    with open(file_path, 'wb') as file:
        # Serialize and save the dictionary using pickle
        pickle.dump(camera_dict, file)



def show_dist():
    with open('camera_pos_dict.pkl', 'rb') as file:
        data = pickle.load(file)
        print(data)


if __name__ == '__main__':
    images = load_images()
    _pattern = checkered_board
    process_video(_pattern)
    # show_dist()


    # for key in images:
    #     image_name = key # for example : cam6.jpg
    #     image = images[image_name]
    #     x, y, z = get_global_camera_position(image,image_name, _pattern)





