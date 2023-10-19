import math
import os
import json
from scipy.spatial.transform import Rotation
import numpy as np
import pickle
from evaluation import rotation_matrix_to_euler_angles
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6
from camera_calibration.vanishing_points import pattern_corner_detect, all_lines_detect, parameter_lines_detect, \
    line_fit_ransac, intersection_2Dpoints_detect, intrinsic_parameters, principal_point_coordinates, rotation, \
    translation, get_distance_to_calibration_pattern, get_intrinsic_matrix, focal_length_calc
from load import load_images
import cv2
import matplotlib.lines as mlines



def get_franz_data_extrinsic():
    base_path = "franz_data"
    cameras = ["mad-museum1", "mad-museum2", "mad-museum3", "mad-museum4", "mad-museum5", "mad-museum6"]
    names_nina = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
    combined_data = {}

    for camera, nina_name in zip(cameras, names_nina):
        file_path = os.path.join(base_path, camera,
                                 f"extrinsic/calibration.json")
        with open(file_path, "r") as file:
            data = json.load(file)
            rotation = data["rotation"]
            translation = data["translation"]

            if nina_name not in combined_data:
                combined_data[nina_name] = {}
            combined_data[nina_name]['rotation'] = rotation
            combined_data[nina_name]['translation'] = translation


    # Save the combined data to a new JSON file
    output_file_path = "franz_data/combined_Franz_data.json"
    with open(output_file_path, "w") as output_file:
        json.dump(combined_data, output_file, indent=4)

    print(f"Combined data saved to {output_file_path}")

def get_franz_euler_angles():
    # Load the combined JSON data
    combined_file_path = "franz_data/combined_Franz_data.json"
    with open(combined_file_path, "r") as combined_file:
        combined_data = json.load(combined_file)

    # Loop through the combined data and access rotations and translations
    for camera_name in combined_data:
        rotation_vector = combined_data[camera_name]['rotation']

        r = Rotation.from_quat(rotation_vector).as_matrix().T
        # print("Rotation matrix:", r)

        angles = rotation_matrix_to_euler_angles(r)
        print(camera_name, "angles", angles)
        print("-" * 20)  # Separator between camera data
        #
        combined_data[camera_name]["rotation_matrix"] = r.tolist()
        combined_data[camera_name]["euler_angles"] = angles.tolist()


    # Save the updated combined data back to the JSON file
    with open(combined_file_path, "w") as combined_file:
            json.dump(combined_data, combined_file, indent=4)

    print("Euler angles updated in the JSON data.")

def compare_rotation():
    print("ROTATION ")
    with open('nina_data/camera_av_parameters.json', 'r') as file:
        nina_data = json.load(file)
    with open('franz_data/combined_Franz_data.json', 'r') as file:
        franz_data = json.load(file)

        # Accessing individual items within the dictionary
        for camera in nina_data:
            nina_av_angles = np.array(nina_data[camera]["av_angles"])
            nina_std_angles= np.array(nina_data[camera]["std_angles"])
            franz_angles = np.array(franz_data[camera]["euler_angles"])
            abs_difference = np.abs(franz_angles - nina_av_angles)
            # epsilon = 1e-6
            # relative_difference = (franz_angles - nina_av_angles) / (nina_av_angles + epsilon)
            #
            # z_score =(franz_angles - nina_av_angles)/nina_std_angles
            print(camera)
            dif = franz_angles - nina_av_angles

            # print("abs_difference", [round(coord, 2) for coord in abs_difference])
            # print("relative_difference", relative_difference)
            # print("z_score", z_score, "\n")
            print("-" * 20)  # Separator between camera data


def get_franz_coordinates():
    # Load the combined JSON data
    combined_file_path = "franz_data/combined_Franz_data.json"
    with open(combined_file_path, "r") as combined_file:
        combined_data = json.load(combined_file)

    # Loop through the combined data and access rotations and translations
    for camera_name in combined_data:
        rotation_vector = combined_data[camera_name]['rotation']
        translation_vector = combined_data[camera_name]['translation']
        r = Rotation.from_quat(rotation_vector).as_matrix().T
        c = np.matmul(-r, np.array(translation_vector))
        combined_data[camera_name]["coordinates"] = c.tolist()


    # Save the updated combined data back to the JSON file
    with open(combined_file_path, "w") as combined_file:
        json.dump(combined_data, combined_file, indent=4)

    print("Euler angles updated in the JSON data.")


def compare_coordinates():
    print("COORDINATES")
    with open('nina_data/camera_av_parameters.json', 'r') as file:
        nina_data = json.load(file)
    with open('franz_data/combined_Franz_data.json', 'r') as file:
        franz_data = json.load(file)

        # Accessing individual items within the dictionary
        for camera in nina_data:
            nina_av_positions = np.array(nina_data[camera]["av_positions"])
            franz_positions = np.array(franz_data[camera]["coordinates"]).flatten()*1000
            abs_difference = np.abs(franz_positions - nina_av_positions)
            dif= franz_positions[:2] - nina_av_positions[:2]
            magnitude = math.sqrt(dif[0]**2 + dif[1]**2)
            print("camera", camera)
            print("magnitude", round(magnitude, 2) )

            print("abs_difference round", [round(coord, 2) for coord in abs_difference], "\n")

def plot_coordinates():

    with open('nina_data/camera_av_parameters.json', 'r') as file:
        nina_data = json.load(file)
    with open('franz_data/combined_Franz_data.json', 'r') as file:
        franz_data = json.load(file)

        # Create a figure and axis
        fig, ax = plt.subplots()

        #Accessing individual items within the dictionary
        for camera in nina_data:
            nina_av_positions = np.array(nina_data[camera]["av_positions"])[:2]
            franz_positions = np.array(franz_data[camera]["coordinates"]).flatten()[:2]*1000

            ax.plot(nina_av_positions[0], nina_av_positions[1], 'r+')
            ax.plot(franz_positions[0], franz_positions[1], 'bo')
            plt.text(franz_positions[0], franz_positions[1]-200, camera, fontsize=15, ha='center', va='top')


        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        # Show the plot
        plt.gca().grid("on")
        plt.gca().set_aspect('equal', adjustable='box')
        # Create proxy artists for the legend
        nina_proxy = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=10, label='Vanishing points method')
        franz_proxy = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10,
                                    label='Validation data')
        # Add a legend with the proxy artists
        plt.legend(handles=[nina_proxy, franz_proxy], bbox_to_anchor=(1, 1), loc = 'upper left')

        # Adjust figure size and margins
        fig.set_size_inches(10,10)  # Adjust the figure size as needed

        plt.savefig('coord_plot', dpi=300, bbox_inches='tight')
        plt.show()


def image_for_report_1():
    images = load_images()
    pattern = checkered_board
    image_name = "cam3.jpg"
    test_image = images[image_name]

    # Detect corners
    p0 = principal_point_coordinates(test_image)
    corners = pattern_corner_detect(test_image, pattern["dimension"]) - p0

    horizontal_lines, vertical_lines = all_lines_detect(pattern["dimension"], corners)
    horizontal_lines_parameters = [line_fit_ransac(test_image, line) for line in horizontal_lines]
    vertical_lines_parameters = [line_fit_ransac(test_image, line) for line in vertical_lines]

    perimeter_lines = parameter_lines_detect(test_image, pattern["dimension"], corners)
    fitted_lines_parameters = [line_fit_ransac(test_image, line) for line in perimeter_lines]

    camera_name = image_name.split('.')[0]
    starting_pair = globals()[camera_name]["start_pair"]
    u_pair = globals()[camera_name]["u_pair"]
    v_pair = globals()[camera_name]["v_pair"]


    # Computing Vanishing points
    vp_u = intersection_2Dpoints_detect(fitted_lines_parameters[u_pair[0]], fitted_lines_parameters[u_pair[1]])
    # vertical intersection point
    vp_v = intersection_2Dpoints_detect(fitted_lines_parameters[v_pair[0]], fitted_lines_parameters[v_pair[1]])
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    plt.imshow(test_image)

    plt.scatter(*np.array([vp_u+p0]).T, marker="*", c="b" )
    # Annotate the point with its name
    # plt.annotate("Fu", (vp_u[0]+p0[0], vp_u[1]+p0[1]), textcoords="offset points", xytext=(0, 10), ha='center',
    #              fontsize=12, color='black')
    plt.scatter(*np.array([vp_v+p0]).T, marker="*", c="b")
    # plt.annotate("Fv", (vp_v[0]+p0[0], vp_v[1]+p0[1]), textcoords="offset points", xytext=(0, 10), ha='center',
    #              fontsize=12, color='black')
    # plt.gca().set_ylim(+p0[1], -p0[1])

    A_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[3])
    B_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[1], fitted_lines_parameters[3])
    C_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[1], fitted_lines_parameters[2])
    D_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[2])
    plt.scatter(*np.array([A_point_im+p0]).T, c="r", s = 20)
    plt.scatter(*np.array([B_point_im+p0]).T, c="r", s = 20)
    plt.scatter(*np.array([C_point_im+p0]).T, c="r", s = 20)
    plt.scatter(*np.array([D_point_im+p0]).T, c="r", s = 20)
    # Save the plot with specified DPI for high quality
    plt.savefig('report_image_1', dpi=300, bbox_inches='tight')

    plt.show(block=True)


def get_round():
    with open('nina_data/camera_av_parameters.json', 'r') as file:
        nina_data = json.load(file)

        for camera in nina_data:
            nina_av_positions = np.array(nina_data[camera]["av_positions"])
            std = np.array(nina_data[camera]["std_positions"])
            nina_av_angles = np.array(nina_data[camera]["av_angles"])
            std_angle = np.array(nina_data[camera]["std_angles"])
            print(camera)
            rounded_positions = [round(coord, 2) for coord in nina_av_positions]
            rounded_std = [round(coord, 2) for coord in std]
            rounded_angles = [round(coord, 2) for coord in nina_av_angles]
            rounded_std_angles = [round(coord, 2) for coord in std_angle]

            print("angle: ", rounded_angles)
            print("std", rounded_std_angles)

            print("coord: ", rounded_positions)
            print("std", rounded_std)

            print("-" * 20)  # Separator between camera data

def get_relative_angle():
    combined_file_path = "franz_data/combined_Franz_data.json"
    with open(combined_file_path, "r") as combined_file:
        franz_data = json.load(combined_file)

    combined_file_path = "nina_data/camera_av_parameters.json"
    with open(combined_file_path, "r") as combined_file:
        nina_data = json.load(combined_file)
    for camera_name in nina_data:
        print(camera_name)
        rotation_matrix = franz_data[camera_name]['rotation_matrix']
        reference_view_vector = np.dot(np.transpose(rotation_matrix), np.array([0, 0, 1]))
        reference_view_vector = reference_view_vector / np.linalg.norm(reference_view_vector)

        rotation_matrix = nina_data[camera_name]['extrinsic_matrix']
        vp_view_vector = np.dot(np.transpose(rotation_matrix), np.array([0, 0, 1]))
        vp_view_vector[1]*=-1

        dot_product = np.dot(reference_view_vector, vp_view_vector)
        magnitude_A = np.linalg.norm(reference_view_vector)
        magnitude_B = np.linalg.norm(vp_view_vector)
        cosine_theta = dot_product / (magnitude_A * magnitude_B)
        angle_radians = np.arccos(cosine_theta)
        angle_degrees = np.degrees(angle_radians)
        print(round(angle_degrees, 2), "\n")
    print("\n")

def get_relative_angle_2():
    combined_file_path = "franz_data/combined_Franz_data.json"
    with open(combined_file_path, "r") as combined_file:
        franz_data = json.load(combined_file)

    combined_file_path = "nina_data/camera_av_parameters.json"
    with open(combined_file_path, "r") as combined_file:
        nina_data = json.load(combined_file)
    for camera_name in nina_data:
        print(camera_name)
        rotation_matrix = franz_data[camera_name]['rotation_matrix']
        reference_view_vector = np.dot(np.transpose(rotation_matrix), np.array([0, 0, 1]))
        reference_view_vector = reference_view_vector / np.linalg.norm(reference_view_vector)

        nina_positions = np.array(nina_data[camera_name]["av_positions"])
        franz_positions = np.array(franz_data[camera_name]["coordinates"]).flatten() * 1000
        vector_franz_nina = nina_positions - franz_positions
        vector_fromfranztonina = vector_franz_nina / np.linalg.norm(vector_franz_nina)

        dot_product = np.dot(reference_view_vector, vector_fromfranztonina)
        magnitude_A = np.linalg.norm(reference_view_vector)
        magnitude_B = np.linalg.norm(vector_fromfranztonina)
        cosine_theta = dot_product / (magnitude_A * magnitude_B)
        angle_radians = np.arccos(cosine_theta)
        angle_degrees = np.degrees(angle_radians)
        print(round(angle_degrees, 2), "\n")




# def get_franz_view_dir():
#     combined_file_path = "franz_data/combined_Franz_data.json"
#     with open(combined_file_path, "r") as combined_file:
#         franz_data = json.load(combined_file)
#     print("New method ")
#     for camera_name in franz_data:
#         print(camera_name)
#         x, y, z, w = franz_data[camera_name]['rotation']
#         view_dir_x =  2 * (x * z + w * y)
#         view_dir_y = 2 * (y * z - w * x)
#         view_dir_z = 1 - 2 * (x * x + y * y)
#         view_dir_vector = np.array([view_dir_x, view_dir_y,view_dir_z ])
#         view_dir_vector = view_dir_vector / np.linalg.norm(view_dir_vector)
#         print(view_dir_vector)
#     print("\n")







if __name__ == '__main__':
    get_relative_angle_2()

