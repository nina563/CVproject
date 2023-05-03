import itertools
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import RANSACRegressor
from sympy import Line3D, Point3D


def pattern_corner_detect(img, pattern_size):
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_scale, pattern_size, None)

    # Refine the corner locations using the cornerSubPix function
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray_scale, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(gray_scale, pattern_size, corners, ret)
    #     if ret == True: # if the detection was succesful or not
    # #@Draw and display the corners
    #         cv2.drawChessboardCorners(gray_scale, pattern_size, corners, ret)
    #         cv2.imshow("", gray_scale)
    #         cv2.waitKey(0)

    # reform corners arry from (n,1,2) into (n,2)
    reshaped_corners = corners.reshape(-1, 2)
    return reshaped_corners


def edge_corners_detect(img, pattern_size, corners):
    # generating ids of the spesific pattern
    x, y = pattern_size
    corner_ids = [0, x - 1, x * y - x, x * y - 1]
    edge_corners = corners[corner_ids]
    # Draw the edge corners on the image
    for corner in edge_corners:
        x, y = corner.ravel()
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    #
    # # Show the image
    # cv2.imshow('Chessboard Corners', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edge_corners


def all_lines_detect(pattern_size, corners):
    x_size, y_size = pattern_size
    horizontal = []
    vertical = []

    for i in range(y_size):
        line = corners[(i * x_size):(x_size + i * x_size)].copy()
        horizontal.append(line)

    for var_x in range(x_size):
        line = np.array([corners[var_x + i * x_size].copy() for i in range(y_size)])
        vertical.append(line)

    return horizontal, vertical


def parameter_lines_detect(img, pattern_size, corners):
    x, y = pattern_size
    first_horizontal = corners[0:x].copy()
    second_horizontal = corners[x * y - x:x * y].copy()
    first_vertical = np.array([corners[x * i].copy() for i in range(y)])
    second_vertical = np.array([corners[x * i - 1].copy() for i in range(1, y + 1)])

    for corner in first_horizontal:
        x, y = corner.ravel()
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    for corner in second_horizontal:
        x, y = corner.ravel()
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    for corner in first_vertical:
        x, y = corner.ravel()
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), 2)

    for corner in second_vertical:
        x, y = corner.ravel()
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), 2)

    # Show the image
    # cv2.imshow('Chessboard Corners', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return first_horizontal, second_horizontal, first_vertical, second_vertical


def line_fit_ransac(img, line):
    height, width, _ = img.shape
    X = line[:, 0].reshape(-1, 1)
    y = line[:, 1].reshape(-1, 1)
    ransac = RANSACRegressor()
    ransac.fit(X, y)

    # The coefficients of the fitted line
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    parameters = [slope, intercept]

    parameters = np.array(np.squeeze(np.array(parameters)))
    return parameters


def intersection_2Dpoints_detect(line_1_parameter, line_2_parameter):
    slope1, intercept1 = line_1_parameter
    slope2, intercept2 = line_2_parameter
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    y_intersect = slope1 * x_intersect + intercept1
    points = np.array((x_intersect, y_intersect))
    return points


def find_array_of_intersecting_2Dpoints(lines):
    # Find all the intersection points among the lines
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            intersection = intersection_2Dpoints_detect(line1, line2)
            if intersection is not None:
                intersections.append(intersection)
    return intersections


def distance(point1, point2):  # Calculates the distance between two points in a 2D coordinate system.
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def focal_length_calc(vp_u, vp_v, p, p_uv):
    OcVi_len = math.sqrt(distance(vp_v, p_uv) * distance(p_uv, vp_u))
    focal_length = math.sqrt(pow(OcVi_len, 2) - pow(distance(p, p_uv), 2))
    return focal_length


def focal_length(vp_u, vp_v, p=(0, 0)):
    return np.sqrt((-(vp_u - p)).dot(vp_v - p))


def principal_point_coordinates(img):
    height, width, channels = img.shape
    center_x = int(width / 2)
    center_y = int(height / 2)
    return [center_x, center_y]


def intrinsic_parameters(img, vp_u, vp_v):
    p_point = principal_point_coordinates(img)  # u0, v0 for the intrinsic matrix
    u0 = p_point[0]
    v0 = p_point[1]
    skew = 0
    scale_factor_u = scale_factor_v = 1

    # Convert points to numpy arrays
    vp_u = np.array(vp_u)
    vp_v = np.array(vp_v)
    p = np.array(p_point)

    ap = p_point - vp_v
    ab = vp_u - vp_v
    p_uv = vp_v + np.dot(ap, ab) / np.dot(ab, ab) * ab
    f = focal_length_calc(vp_u, vp_v, p, p_uv)  # focal length
    K = np.array([[scale_factor_u * f, skew, u0],
                  [0, scale_factor_v * f, v0],
                  [0, 0, 1]])
    return f, K


def rotation(vp_u, vp_v, f):
    len_u_rc = math.sqrt(vp_u[0] ** 2 + vp_u[1] ** 2 + f ** 2)
    len_v_rc = math.sqrt(vp_v[0] ** 2 + vp_v[1] ** 2 + f ** 2)
    u_rc = np.array([vp_u[0], vp_u[1], f]) / len_u_rc
    v_rc = np.array([vp_v[0], vp_v[1], f]) / len_v_rc

    w_rc = np.cross(u_rc, v_rc)

    u_rc = u_rc.reshape(3, 1)
    v_rc = v_rc.reshape(3, 1)
    w_rc = w_rc.reshape(3, 1)

    # rotation from world to camera
    rotation = np.hstack((u_rc, v_rc, w_rc))
    return rotation


def construct_plane(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    d = -np.dot(normal, np.array(p1))
    return np.concatenate((normal, [d])), normal


def translation(vp_u, f, principal_point, square_size, A_point, D_point):
    K_pos = 4
    K_point = (np.array(D_point) - np.array(A_point)) * K_pos / 7 + np.array(A_point)

    k_rc = np.array([*(K_point - principal_point), f])
    a_rc = np.array([*(A_point - principal_point), f])
    vp_u_3d = np.array([*vp_u, f])

    # Find a point in the right direction first, and the actual intersection point after.
    Kp_rc = (k_rc - a_rc).dot(vp_u_3d)/np.linalg.norm(vp_u_3d) * vp_u_3d / np.linalg.norm(vp_u_3d) + a_rc
    Kp_rc = intersection_3Dpoints_detect(a_rc, Kp_rc, [0, 0, 0], k_rc)

    AK_ro = square_size * K_pos  # note this is in mm
    OA_ro = norm(a_rc) / norm(Kp_rc - a_rc) * AK_ro

    return OA_ro  # in mm


def project_onto_plane(v, normal, o):
    # Compute the dot product of the normal vector with the vector you want to project.
    signed_distance = np.dot(normal, v - o)
    # Subtract the projection of the vector onto the normal vector from the original vector-  Calculate the projection of vector AP onto the plane OAK.
    projection = v - signed_distance / np.dot(normal, normal) * normal
    return projection


def intersection_3Dpoints_detect(a_rc, p_new, o_rc, k_rc):
    # source https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d
    p1 = np.array(a_rc)
    p2 = np.array(o_rc)
      # d2
    e1 = p_new - a_rc
    e2 = k_rc - o_rc

    # Find the vector connecting the two points
    v = p1 - p2

    # Find the normal vector of the plane
    n = np.cross(e1, e2)

    # Find the distance between the two lines
    dist = np.dot(n, -v) / np.linalg.norm(n)

    if abs(dist) <= 1e-12:
        t1 = np.cross(e2, n).dot(p2 - p1) / n.dot(n)
        t2 = np.cross(e1, n).dot(p2 - p1) / n.dot(n)

        intersection_point = p1 + t1 * e1
        print("The two lines intersect at point", intersection_point)

        assert (intersection_point != p2 + t2 * e2).any(), f"Closest point differs: {intersection_point} != {p2 + t2 * e2}"

    else:
        raise RuntimeError(f"The two lines do not intersect. Separation = {dist}")

    return intersection_point


def combinations(A, B):
    combinations = list(itertools.product(A, B))
    return combinations


def average_extraction(horizontal_lines_parameters, pattern_size, perimeter_lines, vertical_lines_parameters):
    # get an array of intersecting all the horizontal lines among each other
    horizontal_vp = find_array_of_intersecting_2Dpoints(horizontal_lines_parameters)
    vertical_vp = find_array_of_intersecting_2Dpoints(vertical_lines_parameters)
    # VIZUALISATION
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for x,y in horizontal_vp:
    #     ax.scatter([x], [y],  color='b', marker='o')
    # for x,y in vertical_vp:
    #     ax.scatter([x], [y],  color='b', marker='o')
    # ax.scatter(vp_u[0], vp_u[1], color='g', marker='o')
    # ax.scatter(vp_v[0], vp_v[1], color='g', marker='o')
    #
    # for x,y in all_corners:
    #     ax.scatter([x], [y],  color='b', marker='o')
    combinations_vp = combinations(horizontal_vp, vertical_vp)
    min = float('inf')
    count = 0
    for i in combinations_vp:
        vp_u, vp_v = i
        f, K = intrinsic_parameters(img, vp_u, vp_v)
        rotation_matrix = rotation(vp_u, vp_v, f)
        ##perimeter_lines ---- first_horizontal, second_horizontal, first_vertical, second_vertical------([slop, intercept],...)
        # i want to add point vp_u to the first_horizontal, second_horizontal, and vpv to first_vertical, second_vertical
        np.append(perimeter_lines[0], vp_u)
        np.append(perimeter_lines[1], vp_u)
        np.append(perimeter_lines[2], vp_v)
        np.append(perimeter_lines[3], vp_v)
        fitted_lines_parameters = [line_fit_ransac(img, line) for line in perimeter_lines]
        A_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[3])
        D_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[2])
        dist = translation(vp_u, f, img, rotation_matrix, all_corners, square_size_big, pattern_size, A_point_im,
                           D_point_im)
        if dist < min:
            min = dist
        if count == 0:
            break
        count += 1
    print(min)
    plt.show(block=True)


def get_distance_to_calibration_pattern(test_image, pattern):
    """Given an image containing a checkerd pattern, it returns the distance to the left,lower  most corner in the
       pattern with respect to the camera view. The distance is in the units specified by the pattern.square_size."""

    # Detect corners and recenter so that P0 is the new origin.
    p0 = principal_point_coordinates(test_image)
    corners = pattern_corner_detect(test_image, pattern["dimension"]) - p0

    # Use perimeter lines to compute the distance
    perimeter_lines = parameter_lines_detect(test_image, pattern["dimension"], corners)
    fitted_lines_parameters = [line_fit_ransac(test_image, line) for line in perimeter_lines]

    # Computing Vanishing points
    vp_u = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[1])
    vp_v = intersection_2Dpoints_detect(fitted_lines_parameters[2], fitted_lines_parameters[3])

    # Select A and D in camara coordinates based on fitted lines
    A_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[3])
    D_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[2])

    f = focal_length(vp_u, vp_v)

    # we have already shifted everything so that 0,0 is in P0
    OA_ro = translation(vp_u, f, (0, 0), pattern["square_size"],
                        A_point_im,
                        D_point_im)

    # average_extraction(horizontal_lines_parameters, pattern_size, perimeter_lines, vertical_lines_parameters)
    # This position is in mm
    camera_pos = np.matmul(np.linalg.inv(rotation(vp_u, vp_v, f)),
                           (OA_ro * ([*A_point_im, f] / norm([*A_point_im, f]))))

    return -camera_pos

def get_rotation(test_image, pattern):
    """Given an image containing a checkerd pattern, it returns the rotation of the camera."""

    # Detect corners and recenter so that P0 is the new origin.
    p0 = principal_point_coordinates(test_image)
    corners = pattern_corner_detect(test_image, pattern["dimension"]) - p0

    # Use perimeter lines to compute the distance
    perimeter_lines = parameter_lines_detect(test_image, pattern["dimension"], corners)
    fitted_lines_parameters = [line_fit_ransac(test_image, line) for line in perimeter_lines]

    # Computing Vanishing points
    vp_u = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[1]) # in camera coordinates
    vp_v = intersection_2Dpoints_detect(fitted_lines_parameters[2], fitted_lines_parameters[3])

    f = focal_length(vp_u, vp_v)
    rotation_matrix = rotation(vp_u, vp_v, f)
    return rotation_matrix
