import matplotlib.pyplot as plt
import matplotlib

import cv2
import numpy as np

from callibration_patterns import checkered_board
from camera_calibration.vanishing_points import pattern_corner_detect, all_lines_detect, parameter_lines_detect, \
    line_fit_ransac, intersection_2Dpoints_detect, intrinsic_parameters, principal_point_coordinates, rotation, \
    translation, focal_length, get_distance_to_calibration_pattern


def visual_test(test_image, pattern):
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

    # Computing Vanishing points
    vp_u = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[1])

    # vertical intersection point
    vp_v = intersection_2Dpoints_detect(fitted_lines_parameters[2], fitted_lines_parameters[3])

    draw_image_with_corners(corners, test_image, "Vanishing Points", p0)
    plt.scatter(*np.array([vp_u, vp_v]).T, marker="*")
    compare_lines_with_parameters(fitted_lines_parameters, perimeter_lines)
    plt.gca().set_ylim(+p0[1], -p0[1])

    A_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[3])
    D_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[2])
    plt.scatter(*np.array([A_point_im, D_point_im]).T, marker="*", c="r")


    f = focal_length(vp_u, vp_v)
    print(f"Focal length: {f} pixels")

    rotation_matrix = rotation(vp_u, vp_v, f)

    # we have already shifted everything so that 0,0 is in P0
    OA_ro = translation(vp_u, f, (0, 0), pattern["square_size"],
                       A_point_im,
                       D_point_im)

    print(f"Distance to pattern orifgin A: {OA_ro / 1000}")

    return  OA_ro



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
    _test_image_path = "frame4.jpg"
    _test_image = cv2.imread(_test_image_path)
    _pattern = checkered_board

    vt_oa_ro = visual_test(_test_image, _pattern)
    assert(vt_oa_ro == get_distance_to_calibration_pattern(_test_image, _pattern))


    plt.show(block=True)


