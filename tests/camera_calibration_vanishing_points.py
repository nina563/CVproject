import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy.linalg import norm
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6
from camera_calibration.vanishing_points import pattern_corner_detect, all_lines_detect, parameter_lines_detect, \
    line_fit_ransac, intersection_2Dpoints_detect, intrinsic_parameters, principal_point_coordinates, rotation, \
    translation, focal_length, get_distance_to_calibration_pattern
from load import load_images


def visual_test(test_image,image_name, pattern):
    # Detect corners
    p0 = principal_point_coordinates(test_image)
    corners = pattern_corner_detect(test_image, pattern["dimension"]) - p0
    # draw_image_with_corners(corners, test_image, p0=p0)

    # all vertical and horizontal lines
    horizontal_lines, vertical_lines = all_lines_detect(pattern["dimension"], corners)
    horizontal_lines_parameters = [line_fit_ransac(test_image, line) for line in horizontal_lines]
    # draw_image_with_corners(corners, test_image)
    # compare_lines_with_parameters(corners, horizontal_lines_parameters, horizontal_lines)

    vertical_lines_parameters = [line_fit_ransac(test_image, line) for line in vertical_lines]
    # draw_image_with_corners(corners, test_image)
    # compare_lines_with_parameters(corners, vertical_lines_parameters, vertical_lines)

    perimeter_lines = parameter_lines_detect(test_image, pattern["dimension"], corners)
    fitted_lines_parameters = [line_fit_ransac(test_image, line) for line in perimeter_lines]
    # draw_image_with_corners(corners, test_image)
    # compare_lines_with_parameters(corners, fitted_lines_parameters, perimeter_lines)

    camera_name = image_name.split('.')[0]
    starting_pair = globals()[camera_name]["start_pair"]
    u_pair = globals()[camera_name]["u_pair"]
    v_pair = globals()[camera_name]["v_pair"]

    first = starting_pair[0]
    print("camera_name", camera_name,"u", u_pair, "v", v_pair )


    # Computing Vanishing points
    vp_u = intersection_2Dpoints_detect(fitted_lines_parameters[u_pair[0]], fitted_lines_parameters[u_pair[1]])

    # vertical intersection point
    vp_v = intersection_2Dpoints_detect(fitted_lines_parameters[v_pair[0]], fitted_lines_parameters[v_pair[1]])

    draw_image_with_corners(corners, test_image, "Vanishing Points", p0)
    plt.scatter(*np.array([vp_u]).T, marker="*")
    # compare_lines_with_parameters(fitted_lines_parameters, perimeter_lines)
    plt.gca().set_ylim(+p0[1], -p0[1])

    # Select A and D in camara coordinates based on fitted lines
    A_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[3])
    B_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[1], fitted_lines_parameters[3])
    C_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[1], fitted_lines_parameters[2])
    D_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[2])


# choosing the start coordinates for the calculations
    if first == "A" :
        start_coordinate_system = A_point_im
        second_important_point = D_point_im
    elif first =="B":
        start_coordinate_system = B_point_im
        second_important_point = A_point_im
    elif first =="C":
        start_coordinate_system = C_point_im
        second_important_point = B_point_im
    elif first=="D":
        start_coordinate_system = D_point_im
        second_important_point = C_point_im


    plt.scatter(*np.array([start_coordinate_system, second_important_point]).T, marker="*", c="r")
    f = focal_length(vp_u, vp_v)

    print(f"Focal length: {f} pixels")

    rotation_matrix = rotation(vp_u, vp_v, f)

    #we have already shifted everything so that 0,0 is in P0
    OA_ro = translation(vp_u, f, (0, 0), pattern["square_size"],
                        start_coordinate_system,
                        second_important_point, camera_name)
    camera_pos = np.matmul(np.linalg.inv(rotation(vp_u, vp_v, f)),
                           (OA_ro * ([*start_coordinate_system, f] / norm([*start_coordinate_system, f]))))

    return OA_ro


def draw_image_with_corners(corners, test_image, title="", p0=(0, 0)):
    plt.figure()
    plt.suptitle(title)
    extent = np.array([0, test_image.shape[1], test_image.shape[0], 0])
    delta = [p0[0]] * 2 + [p0[1]] * 2
    plt.imshow(test_image, extent=extent - delta)
    plt.scatter(*corners.T)


def compare_lines_with_parameters(fitted_lines_parameters, lines):
    for param, line in zip(fitted_lines_parameters, lines):
        x = np.arange(*plt.gca().get_xlim(), 10)
        y = x * param[0] + param[1]
        plt.plot(x, y, c="b", lw=0.5)
        plt.plot(*line.T)


if __name__ == '__main__':
    matplotlib.use('QtCairo')
    images = load_images()
    _pattern = checkered_board

    for key in images:
        image_name = key # for example : cam6.jpg
        image = images[image_name]
        vt_oa_ro = visual_test(image, image_name, _pattern)
        x, y, z = get_distance_to_calibration_pattern(image,image_name, _pattern)
        assert ((vt_oa_ro - np.linalg.norm([x, y, z])) < 1e-12)

        print(f"Distance to the wall: {y/1000:0.2f}m, "
              f"distance from the focal axis: {x/1000:0.2f}m, "
              f"camera installation height: {z/1000:0.2f}m")
        print("\n")

    plt.show(block=True)
