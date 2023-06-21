from scene_calibration import floor_map, get_global_transform_per_camera, rotation_matrix,\
    get_coordinates_from_point_names, get_extrinsic_matrix_per_camera, get_local_to_global_transform, get_rotation
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6

from camera_calibration.vanishing_points import get_distance_to_calibration_pattern, get_intrinsic_matrix, pattern_corner_detect, principal_point_coordinates
import numpy as np
from load import load_images
from math import asin, pi, atan2, cos
import cv2
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


def get_global_camera_position(test_image,image_name, pattern):
    camera_name = image_name.split('.')[0]
    position = globals()[camera_name]["position"]  # 1,2,3 position on the floor
    starting_pair = globals()[camera_name]["start_pair"]
    letter_start_coordinate_system = starting_pair[0]

    # name of the point on the floor , that is our start of coordinate system
    point_name = letter_start_coordinate_system + "_pattern_coordinate_" + str(position) + "_position"

    transform_from_local_to_global = get_global_transform_per_camera(point_name, pattern)
    print("letter_start_coordinate_system", letter_start_coordinate_system)
    rotation_local_to_global = rotation_matrix(letter_start_coordinate_system)
    local_camera_pos = get_distance_to_calibration_pattern(test_image,image_name, pattern)# in mm
    # print("local_camera_pos",local_camera_pos)
    reshaped_array = np.append(local_camera_pos, 1)
    # Reshape the modified array to shape (4, 1)
    reshaped_local_camera_pos = np.reshape(reshaped_array, (4, 1))

    global_local_camera = np.matmul(transform_from_local_to_global,np.matmul(rotation_local_to_global, reshaped_local_camera_pos))
    # print("global_local_camera", global_local_camera)
    global_local_camera= global_local_camera/global_local_camera[-1]
    # print("global_local_camera", global_local_camera)
    # print("global", np.squeeze(global_local_camera[:3]))
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
            index += 1

    file_path = "camera_pos_dict.pkl"

    # Open the file in write mode
    with open(file_path, 'wb') as file:
        # Serialize and save the dictionary using pickle
        pickle.dump(camera_dict, file)


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


    with open('camera_pos_dict.pkl', 'rb') as file:
        dictionary= pickle.load(file)
        for key, coordinates in dictionary.items():
            # Extract x and y coordinates from the list of tuples
            x_values = [coord[0] for coord in coordinates]
            y_values = [coord[1] for coord in coordinates]
            z_values = [coord[2] for coord in coordinates]

            # Plot the points for each camera
            plt.scatter(x_values, y_values, label=key)
        #
        # plotting coordinates of just one camera
        # coord_cam1 = dictionary["cam1"]
        # for coordinate in coord_cam1:
        #     x_values = coordinate[0]
        #     y_values = coordinate[1]
        #     plt.scatter(x_values, y_values)

    # Add labels and legend to the plot
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Show the plot
    plt.gca().grid("on")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def pixel_coord_camN_to_global_coord(camN_image, image_name, pattern):
    x, y = pattern["dimension"]
    corner_ids = [0, x - 1, x * y - x, x * y - 1]
    corners = pattern_corner_detect(camN_image, pattern["dimension"])
    edge_corners = corners[corner_ids] # a,b,c,d
    print("edge_corners", edge_corners)

    extrinsic_matrix = get_extrinsic_matrix_per_camera(camN_image, image_name, pattern)
    intrinsic_matrix = get_intrinsic_matrix(camN_image, image_name, pattern)
    print("intrinsic_matrix", intrinsic_matrix)
    inverse_intrinsic = np.linalg.inv(intrinsic_matrix)
    print("inv", inverse_intrinsic)
    inverse_extrinsic = np.linalg.inv(extrinsic_matrix)
    rotation_local_to_global, transform_from_local_to_global= get_local_to_global_transform(image_name, pattern)

    plt.imshow(camN_image)
    plt.scatter(*edge_corners.T)
    global_coordinates_edge_corners = []


    for point in edge_corners:
        pixel_coordinate =np.append(point, 1)
        print("pixel_coordinate", pixel_coordinate)
        # print("pixel_coordinate", pixel_coordinate)
        camera_coord = np.matmul(inverse_intrinsic, pixel_coordinate )
        # print("camera_coord", camera_coord)
        camera_coord = np.append(camera_coord, 1)
        print("camera_coord", camera_coord)
        global_coord_local =  np.matmul(inverse_extrinsic, camera_coord)
        print("global_coord before transformation", global_coord_local)
        global_coord_global= np.matmul(transform_from_local_to_global,
                                            np.matmul(rotation_local_to_global,global_coord_local))
        print("global_coord after transformation", global_coord_global)
        print("\n")
        global_coordinates_edge_corners.append(global_coord_global)
    print("global_coordinates_edge_corners", global_coordinates_edge_corners)
    # for point in global_coordinates_edge_corners:
    # Extracting x, y, and z coordinates from the array
    x = [point[0] for point in global_coordinates_edge_corners]
    y = [point[1] for point in global_coordinates_edge_corners]
    z = [point[2] for point in global_coordinates_edge_corners]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x, y, z)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def global_coord_to_pixel(image, image_name, pattern):
    extrinsic_matrix = get_extrinsic_matrix_per_camera(image, image_name, pattern)
    intrinsic_matrix = get_intrinsic_matrix(image, image_name, pattern)
    rotation_local_to_global, transform_from_local_to_global = get_local_to_global_transform(image_name, pattern)
    inverse_rotation = np.linalg.inv(rotation_local_to_global)
    inverse_transformation= np.linalg.inv(transform_from_local_to_global)


    array_of_global_points= floor_map(pattern)
    A_pattern_coordinate_3_position = array_of_global_points[0]
    B_pattern_coordinate_3_position= array_of_global_points[1]
    C_pattern_coordinate_3_position= array_of_global_points[2]
    D_pattern_coordinate_3_position = array_of_global_points[3]
    print("A_pattern_coordinate_3_position" ,A_pattern_coordinate_3_position  )
    global_coord_global_A = np.append(A_pattern_coordinate_3_position, 1)
    global_coord_global_B = np.append(B_pattern_coordinate_3_position, 1)
    global_coord_global_C = np.append(C_pattern_coordinate_3_position, 1)
    global_coord_global_D = np.append(D_pattern_coordinate_3_position, 1)

    global_coord_local_A = np.matmul(inverse_rotation, np.matmul(inverse_transformation, global_coord_global_A))
    global_coord_local_B = np.matmul(inverse_rotation, np.matmul(inverse_transformation, global_coord_global_B))
    global_coord_local_C = np.matmul(inverse_rotation, np.matmul(inverse_transformation, global_coord_global_C))
    global_coord_local_D = np.matmul(inverse_rotation, np.matmul(inverse_transformation, global_coord_global_D))

    print("global_coord A ", global_coord_local_A)
    print("global_coord B ", global_coord_local_B)
    print("global_coord C ", global_coord_local_C)
    print("global_coord D ", global_coord_local_D)
    camera_coord_A = np.matmul(extrinsic_matrix, global_coord_local_A)
    camera_coord_B = np.matmul(extrinsic_matrix, global_coord_local_B)
    camera_coord_C = np.matmul(extrinsic_matrix, global_coord_local_C)
    camera_coord_D = np.matmul(extrinsic_matrix, global_coord_local_D)
    print("camera_coord_A", camera_coord_A, "\ncamera_coord_B", camera_coord_B,"\ncamera_coord_C", camera_coord_C,"\ncamera_coord_D", camera_coord_D)

    pixel_coord_A = np.matmul(intrinsic_matrix, camera_coord_A[:3])
    pixel_coord_B = np.matmul(intrinsic_matrix, camera_coord_B[:3])
    pixel_coord_C = np.matmul(intrinsic_matrix, camera_coord_C[:3])
    pixel_coord_D = np.matmul(intrinsic_matrix, camera_coord_D[:3])

    coord_A = pixel_coord_A/pixel_coord_A[-1]
    coord_B = pixel_coord_B/pixel_coord_B[-1]
    coord_C = pixel_coord_C/pixel_coord_C[-1]
    coord_D = pixel_coord_D/pixel_coord_D[-1]
    plt.imshow(image)
    plt.scatter(coord_A[0], coord_A[1], marker="*")
    plt.scatter(coord_B[0], coord_B[1], c = "r")
    plt.scatter(coord_C[0], coord_C[1],  c = "g")
    plt.scatter(coord_D[0], coord_D[1], c = "b")
    plt.show(block=True)

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
    image_name = 'cam1.jpg'
    image = images[image_name]
    # extrinsic = get_extrinsic_matrix_per_camera(image, image_name, _pattern)
    # print("extrinsic\n", extrinsic)

    rotation = get_rotation(image, image_name, _pattern)
    inverse = np.linalg.inv(rotation)
    p = np.array([0,0, 1484.0586271675822])
    local_p = np.matmul( inverse, p)
    print("local_p ",local_p )

    camera_name = image_name.split('.')[0]
    position = globals()[camera_name]["position"]  # 1,2,3 position on the floor
    starting_pair = globals()[camera_name]["start_pair"]
    letter_start_coordinate_system = starting_pair[0]

    # name of the point on the floor , that is our start of coordinate system
    point_name = letter_start_coordinate_system + "_pattern_coordinate_" + str(position) + "_position"

    transform_from_local_to_global = get_global_transform_per_camera(point_name, _pattern)
    print("letter_start_coordinate_system", letter_start_coordinate_system)
    rotation_local_to_global = rotation_matrix(letter_start_coordinate_system)
    # print("local_camera_pos",local_camera_pos)
    reshaped_array = np.append(local_p, 1)
    # Reshape the modified array to shape (4, 1)
    reshaped_local_p_pos = np.reshape(reshaped_array, (4, 1))

    global_p_pos = np.matmul(transform_from_local_to_global,np.matmul(rotation_local_to_global, reshaped_local_p_pos))
    # print("global_local_camera", global_local_camera)
    global_p_pos = global_p_pos /global_p_pos [-1]
    # print("global_local_camera", global_local_camera)
    # print("global", np.squeeze(global_local_camera[:3]))
    print("global p pos ", np.squeeze(global_p_pos [:3]))



    # angles = rotation_matrix_to_euler_angles(rotation_matrix)
    # print("angles", angles)

    cam_1_pos_global = get_global_camera_position(image, image_name, _pattern)
    print("cam_1_pos_global", cam_1_pos_global )

    direction_vector = cam_1_pos_global - global_p_pos
    print("direction_vector", direction_vector)

    x,y,z = direction_vector
    x_angle = np.arctan2(np.sqrt(y*y+z*z), x)*180/pi
    y_angle = np.arctan2(np.sqrt(x*x+z*z), y)*180/pi
    z_angle = np.arctan2(np.sqrt(x * x + y * y), z)*180/pi
    print("x_angle", x_angle)
    print("y_angle", y_angle)
    print("z_angle", z_angle)
    # image_name = 'cam2.jpg'
    # image = images[image_name]
    # cam_pos_2 = get_global_camera_position(image, image_name, _pattern)
    # print("2", cam_pos_2)

    # global_coord_to_pixel(image, image_name, _pattern)

    # for key in images:
    #     image_name = key # for example : cam6.jpg
    #     image = images[image_name]
    #     x, y, z = get_global_camera_position(image,image_name, _pattern)

    # pixel_coord_camN_to_global_coord(image,image_name, _pattern)
    # plt.show(block=True)



    # process_video(_pattern)
    # plot_dict(_pattern)






