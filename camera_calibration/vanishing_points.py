import math

import cv2
import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import RANSACRegressor
from callibration_patterns import checkered_board, cam1, cam2, cam3, cam4, cam5, cam6


def pattern_corner_detect(img, pattern_size):
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_scale, pattern_size, None)

    # Refine the corner locations using the cornerSubPix function
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray_scale, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(gray_scale, pattern_size, corners, ret)
    reshaped_corners = corners.reshape(-1, 2)
    return reshaped_corners


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


def distance(point1, point2):  # Calculates the distance between two points in a 2D coordinate system.
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def focal_length_calc(vp_u, vp_v, p_point): # p - is not (0,0)
    ap = p_point - vp_v
    ab = vp_u - vp_v
    p_uv = vp_v + np.dot(ap, ab) / np.dot(ab, ab) * ab
    OcVi_len = math.sqrt(distance(vp_v, p_uv) * distance(p_uv, vp_u))
    focal_length = math.sqrt(pow(OcVi_len, 2) - pow(distance(p_point, p_uv), 2))
    return focal_length


# def focal_length(vp_u, vp_v, p=(0, 0)):
#     return np.sqrt((-(vp_u - p)).dot(vp_v - p))


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
    f = focal_length_calc(vp_u+p, vp_v+p, p)
    K = np.array([[scale_factor_u * f, skew, u0],
                  [0, scale_factor_v * f, v0],
                  [0, 0, 1]])
    return K


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


def translation(vp_u, f, principal_point, square_size, A_point, D_point, camera_name):
    if camera_name == "cam1" or camera_name == "cam4" or camera_name == "cam6":
        K_pos = 2
        K_point = (np.array(D_point) - np.array(A_point)) * K_pos / 4 + np.array(A_point)
    else:
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
        #print("The two lines intersect at point", intersection_point)
        assert np.allclose(intersection_point,
                                p2 + t2 * e2, rtol=0, atol=1e-06), f"Closest point differs: {intersection_point} != {p2 + t2 * e2}"
    else:
        raise RuntimeError(f"The two lines do not intersect. Separation = {dist}")

    return intersection_point



def get_distance_to_calibration_pattern(test_image,image_name, pattern):
    """Given an image containing a checkerd pattern, it returns the distance to the left,lower  most corner in the
       pattern with respect to the camera view. The distance is in the units specified by the pattern.square_size."""

    # Detect corners and recenter so that P0 is the new origin.
    p0 = principal_point_coordinates(test_image)
    corners = pattern_corner_detect(test_image, pattern["dimension"]) - p0

    # Use perimeter lines to compute the distance
    perimeter_lines = parameter_lines_detect(test_image, pattern["dimension"], corners)
    fitted_lines_parameters = [line_fit_ransac(test_image, line) for line in perimeter_lines]

    camera_name = image_name.split('.')[0]
    starting_pair = globals()[camera_name]["start_pair"]
    u_pair = globals()[camera_name]["u_pair"]
    v_pair = globals()[camera_name]["v_pair"]

    first = starting_pair[0]


    # Computing Vanishing points
    vp_u = intersection_2Dpoints_detect(fitted_lines_parameters[u_pair[0]], fitted_lines_parameters[u_pair[1]])
    vp_v = intersection_2Dpoints_detect(fitted_lines_parameters[v_pair[0]], fitted_lines_parameters[v_pair[1]])

    f = focal_length_calc(vp_u, vp_v, [0,0])

    A_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[3])
    B_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[1], fitted_lines_parameters[3])
    C_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[1], fitted_lines_parameters[2])
    D_point_im = intersection_2Dpoints_detect(fitted_lines_parameters[0], fitted_lines_parameters[2])

    # choosing the start coordinates for the calculations
    if first == "A":
        start_coordinate_system = A_point_im
        second_important_point = D_point_im
    elif first == "B":
        start_coordinate_system = B_point_im
        second_important_point = A_point_im
    elif first == "C":
        start_coordinate_system = C_point_im
        second_important_point = B_point_im
    elif first == "D":
        start_coordinate_system = D_point_im
        second_important_point = C_point_im

    # we have already shifted everything so that 0,0 is in P0
    OA_ro = translation(vp_u, f, (0, 0), pattern["square_size"],
                        start_coordinate_system,
                        second_important_point, camera_name)

    # This position is in mm
    camera_pos = np.matmul(np.linalg.inv(rotation(vp_u, vp_v, f)),
                           (OA_ro * ([*start_coordinate_system, f] / norm([*start_coordinate_system, f]))))

    return -camera_pos


def get_rotation(test_image, image_name, pattern):
    """Given an image containing a checkerd pattern, it returns the rotation of the camera."""

    # Detect corners and recenter so that P0 is the new origin.
    p0 = principal_point_coordinates(test_image)
    corners = pattern_corner_detect(test_image, pattern["dimension"]) - p0

    # Use perimeter lines to compute the distance
    perimeter_lines = parameter_lines_detect(test_image, pattern["dimension"], corners)
    fitted_lines_parameters = [line_fit_ransac(test_image, line) for line in perimeter_lines]

    camera_name = image_name.split('.')[0]
    u_pair = globals()[camera_name]["u_pair"]
    v_pair = globals()[camera_name]["v_pair"]

    # Computing Vanishing points
    vp_u = intersection_2Dpoints_detect(fitted_lines_parameters[u_pair[0]], fitted_lines_parameters[u_pair[1]])
    vp_v = intersection_2Dpoints_detect(fitted_lines_parameters[v_pair[0]], fitted_lines_parameters[v_pair[1]])
    f = focal_length_calc(vp_u, vp_v, [0,0] )
    rotation_matrix = rotation(vp_u, vp_v, f)
    return rotation_matrix


def get_intrinsic_matrix(test_image, image_name, pattern):
    p0 = principal_point_coordinates(test_image)
    corners = pattern_corner_detect(test_image, pattern["dimension"]) - p0
    perimeter_lines = parameter_lines_detect(test_image, pattern["dimension"], corners)
    fitted_lines_parameters = [line_fit_ransac(test_image, line) for line in perimeter_lines]

    camera_name = image_name.split('.')[0]
    u_pair = globals()[camera_name]["u_pair"]
    v_pair = globals()[camera_name]["v_pair"]

    # Computing Vanishing points
    vp_u = intersection_2Dpoints_detect(fitted_lines_parameters[u_pair[0]], fitted_lines_parameters[u_pair[1]])
    vp_v = intersection_2Dpoints_detect(fitted_lines_parameters[v_pair[0]], fitted_lines_parameters[v_pair[1]])

    intrinsic_matrix = intrinsic_parameters(test_image, vp_u, vp_v)
    return intrinsic_matrix
