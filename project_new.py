import cv2
import os
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def extractImages(pathIn):
    parent_dir = os.path.dirname(pathIn)
    folder = 'frames'  # create a folder to store extracted images
    path = os.path.join(parent_dir, folder)
    vid = cv2.VideoCapture(str(pathIn))
    if not os.path.exists(path):
        os.makedirs(path)

    index = 0
    while (index < 6):
        # Extract images
        ret, frame = vid.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(path, "frame{:d}.jpg".format(index)), frame)

        # next frame
        index += 1


def pattern_corner_detect(pattern_size):
    path = (
       "/Users/nina/Desktop/Project/code/vanishing_points/A0 checkerboard /2023_01_11_22_36_36/mad-museum2/00001-2023_01_11_15_35_50/frames/frame4.jpg")
    #path = (
        #"/Users/nina/Desktop/Project/code/vanishing_points/A0 checkerboard /2023_01_11_22_24_53/mad-museum6/00001-2023_01_11_15_24_06/frames/frame4.jpg")
    img = cv2.imread(path)
    # cv2.imshow("", img)
    # cv2.waitKey(0)
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
    return img, reshaped_corners


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

def all_lines_detect(img, pattern_size, corners):
    x_size, y_size = pattern_size
    horizontal =[]
    vertical = []
    array = np.arange(x_size * y_size).reshape(y_size, x_size)
    # print("pattern_size x y ", pattern_size)
    # print(array)
    for i in range(y_size):
        line = corners[(i*x_size):(x_size+i*x_size)]
        horizontal.append(line)
        # for corner in line:
        #     x, y = corner.ravel()
        #     cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    for var_x in range(x_size):
        line = np.array([corners[var_x+i*x_size] for i in range(y_size)])
        vertical.append(line)
        # for corner in line:
        #     x, y = corner.ravel()
        #     cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)


        # Show the image
    # cv2.imshow('Chessboard Corners', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return horizontal, vertical


def parimeter_lines_detect(img, pattern_size, corners):
    x, y = pattern_size
    first_horizontal = corners[0:x]
    second_horizontal = corners[x * y - x:x * y]
    first_vertical = np.array([corners[x * i] for i in range(y)])
    second_vertical = np.array([corners[x * i - 1] for i in range(1, y + 1)])

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
    #draw the lines on top of the image
    x1 = 0
    y1 = int(slope * x1 + intercept)
    x2 = width - 1
    y2 = int(slope * x2 + intercept)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # Display the image
    # cv2.imshow('Fitted line', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    parameters = np.array(np.squeeze(np.array(parameters)))
    return parameters


def intersection_points_detect(line_1_parameter, line_2_parameter):
    slope1, intercept1 = line_1_parameter
    slope2, intercept2 = line_2_parameter
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    y_intersect = slope1 * x_intersect + intercept1
    points =(x_intersect, y_intersect)
    return points

def find_array_of_intersecting_points(lines):
    # Find all the intersection points among the lines
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            intersection = intersection_points_detect(line1, line2)
            if intersection is not None:
                intersections.append(intersection)
    return intersections


def distance(point1, point2):# Calculates the distance between two points in a 2D coordinate system.
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def focal_length_calc(vp_u,vp_v,p, p_uv):
    OcVi_len = math.sqrt(distance(vp_v, p_uv ) * distance(p_uv ,vp_u))
    focal_length =math.sqrt( pow(OcVi_len, 2) -pow(distance(p, p_uv), 2))
    return focal_length

def principal_point_coordinates(img):
    height, width, channels = img.shape
    center_x = int(width / 2)
    center_y = int(height / 2)
    return [center_x, center_y]

def intrinsic_parameters(img, vp_u, vp_v):
    p_point = principal_point_coordinates(img)  # u0, v0 for the intrinsic matrix
    u0= p_point[0]
    v0= p_point[1]
    skew = 0
    scale_factor_u = scale_factor_v = 1
    # Convert points to numpy arrays
    vp_u = np.array(vp_u)
    vp_v = np.array(vp_v)
    p = np.array(p_point)

    ap = p_point - vp_v
    ab = vp_u - vp_v
    p_uv= vp_v + np.dot(ap, ab) / np.dot(ab, ab) * ab
    f=focal_length_calc(vp_u,vp_v,p, p_uv) # focal length
    K =np.array([[scale_factor_u*f , skew,            u0],
                [0,                scale_factor_v*f, v0],
                [0,                0,                1]])
    return f, K


def rotation(vp_u, vp_v, f):

    len_u_rc = math.sqrt(vp_u[0]**2 + vp_u[1]**2 + f**2)
    len_v_rc = math.sqrt(vp_v[0] ** 2 + vp_v[1] ** 2 + f ** 2)
    u_rc = np.array([vp_u[0], vp_u[1], f]) / len_u_rc
    v_rc = np.array([vp_v[0],vp_v[1], f]) / len_v_rc

    w_rc =np.cross(u_rc, v_rc)

    u_rc= u_rc.reshape(3,1)
    v_rc = v_rc.reshape(3,1)
    w_rc= w_rc.reshape(3,1)

    # rotation from world to camera
    rotation = np.hstack((u_rc, v_rc, w_rc))
    return rotation


def construct_plane(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    d = -np.dot(normal, np.array(p1))
    return np.concatenate((normal, [d])), normal

def translation(vp_u, f, img, r, all_corners, square_size, pattern_size, A_point, D_point  ):
    x, y = pattern_size
    principal_point = principal_point_coordinates(img)  # image coordinates of the principal point

    # k_im = all_corners[int(x/2 -1)] # coordinates of k' in the pixel coordinates, k' lies on the projection line a'd'
    a_im = all_corners[int(x - 1)]
    d_im= all_corners[0]
    k_im_new = (d_im-a_im)*4/7 + a_im
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for x,y in all_corners:
    #     ax.scatter([x], [y],  color='b', marker='o')

    # ax.scatter(a_im[0], a_im[1], color='g', marker='o')
    # ax.scatter(k_im_new[0], k_im_new[1], color='g', marker='o')
    # ax.scatter(d_im[0], d_im[1], color='g', marker='o')
    #
    # ax.scatter(A_point[0], A_point[1], color='r', marker='o')
    # ax.scatter(D_point[0], D_point[1], color='r', marker='o')
    K_point = (np.array(D_point) - np.array(A_point)) * 4 / 7 + np.array(A_point)
    # ax.scatter(K_point[0], K_point[1], color='r', marker='o')

    k_u, k_v = K_point
    p_u, p_v = principal_point
    a_u, a_v = A_point
    k_rc = np.array([ (k_u - p_u), (k_v - p_v), f])
    a_rc = np.array([ (a_u - p_u),(a_v - p_v), f])
    o_rc = np.zeros(3)

    #  each square is 125 mm for the A0 pattern
    l = 4* square_size
    AK_ro = l * np.array([1,0,0]).reshape(3,1)  # l * u where u is a unit vector of the world coordinate system
    # print("AK_ro", AK_ro)
    AK_rc =np.dot(r,AK_ro).reshape(3,) #AP_rc = M_oc*AP_ro ----d line
    # print("AK_rc", AK_rc)
    AK_rc_unit= AK_rc / np.linalg.norm(AK_rc) # unit vector

    # print("AK_rc_unit", AK_rc_unit)
    ovp_u = (np.array([vp_u[0],vp_u[1],f ]))/ np.linalg.norm(np.array([vp_u[0],vp_u[1],f ]))
    # print("ovp_u", ovp_u)

    AK_rc = a_rc+ AK_rc

    # print("a_rc",  a_rc)
    # print("k_rc",  k_rc)
    # print("o_rc", o_rc)
    AKO_rc, n  = construct_plane(a_rc, k_rc, o_rc)
    # print("plane equation", AKO_rc)
    # Normalize the normal vector
    plane_norm_unit = n / np.linalg.norm(n)
    # print("normal unit", plane_norm_unit)
    # print("p_intersection_KO", p_intersection_KO)

    dist = intersection_lines(a_rc, AK_rc , o_rc, k_rc)
    # Find the dot product between the vector and normal vector of the plane
    dot_prod = np.dot((AK_rc), plane_norm_unit)
    # Calculate the magnitudes of the vector and normal vector of the plane
    vec_mag = np.linalg.norm((-AK_rc))
    normal_mag = np.linalg.norm(plane_norm_unit)
    # Calculate the angle between the vector and plane
    angle = np.arccos(dot_prod / (vec_mag * normal_mag))
    # print("angle", angle)
    return dist





# #VISUALIZATION
#     # plotting a surface
#     DATA = np.array([k_rc,a_rc, o_rc])
#     Xs = DATA[:, 0]
#     Ys = DATA[:, 1]
#     Zs = DATA[:, 2]
#     #Create a 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
#
#     #Plot the vector as an arrow
#     ax.quiver(a_rc[0], a_rc[1], a_rc[2], -AK_rc[0], -AK_rc[1], -AK_rc[2], color='r') # ok_rc
#
#     #Plot the points as spheres
#     ax.scatter(a_rc[0], a_rc[1], a_rc[2], color='b', marker='o')
#     ax.scatter(k_rc[0], k_rc[1], k_rc[2], color='b', marker='o')
#     ax.scatter(o_rc[0], o_rc[1], o_rc[2], color='b', marker='o')
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

def project_onto_plane(v, normal,o):
    # Compute the dot product of the normal vector with the vector you want to project.
    signed_distance = np.dot(normal, v - o)
    # Subtract the projection of the vector onto the normal vector from the original vector-  Calculate the projection of vector AP onto the plane OAK.
    projection = v - signed_distance / np.dot(normal, normal) * normal
    return projection

def intersection_lines(a_rc, p_new, o_rc, k_rc):
    intersection = None
    # Define the two 3D lines
    # Line 1: x = 1 + t, y = 2 + 2t, z = 3 + 3t
    # Line 2: x = 2 + s, y = 1 + s, z = 4 + 2s
    p1 = np.array(p_new)
    d1 = np.array(a_rc-p_new)
    p2 = np.array(k_rc)
    d2 = np.array(o_rc-k_rc)

    # Find the vector connecting the two points
    v = p1 - p2

    # Find the normal vector of the plane
    n = np.cross(d1, d2)
    # print("cross product", np.cross(p1,p2))
    # print("norm of d1 d2 ",n /np.linalg.norm(n))
    # Find the distance between the two lines
    dist = np.dot(n,-v) / np.linalg.norm(n)
    # print(f"dist {dist}")
    # # If the distance is zero, the two lines intersect
    # if dist == 0:
    #     # Find the parameter t for Line 1
    #     t = np.dot(v, np.cross(p2 - p1, d2)) / np.linalg.norm(np.cross(d1, d2)) ** 2
    #     # Find the point of intersection
    #     intersection = p1 + t * d1
    #     print("The two lines intersect at point", intersection)
    # else:
    #     print("The two lines do not intersect.")
    return dist




def combinations(A,B):
    combinations = list(itertools.product(A, B))
    return combinations

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('MacOSX')
    filelist = list(Path('.').glob('**/*.mp4'))
    pattern_size = (8, 5)
    square_size_big = 125
    img, all_corners = pattern_corner_detect(pattern_size)

    #all vertical and horizontal lines
    horizontal_lines, vertical_lines = all_lines_detect(img, pattern_size, all_corners)
    perimeter_lines = parimeter_lines_detect(img, pattern_size, all_corners)

    fitted_lines_parameters = [line_fit_ransac(img,line) for line in perimeter_lines]# ([slop, intercept],...) #first_horizontal, second_horizontal, first_vertical, second_vertical
    horizontal_lines_parameters =[line_fit_ransac(img,line) for line in horizontal_lines]
    vertical_lines_parameters = [line_fit_ransac(img, line) for line in vertical_lines]
    #horizontal intersection point
    vp_u= intersection_points_detect(fitted_lines_parameters[0], fitted_lines_parameters[1])
    #vertical intersection point
    vp_v= intersection_points_detect(fitted_lines_parameters[2], fitted_lines_parameters[3])



    #get an array of intersecting all the horizontal lines among each other
    horizontal_vp = find_array_of_intersecting_points(horizontal_lines_parameters)
    vertical_vp = find_array_of_intersecting_points(vertical_lines_parameters)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x,y in horizontal_vp:
        ax.scatter([x], [y],  color='b', marker='o')
    for x,y in vertical_vp:
        ax.scatter([x], [y],  color='b', marker='o')
    ax.scatter(vp_u[0], vp_u[1], color='g', marker='o')
    ax.scatter(vp_v[0], vp_v[1], color='g', marker='o')

    for x,y in all_corners:
        ax.scatter([x], [y],  color='b', marker='o')
    combinations_vp = combinations(horizontal_vp,vertical_vp)



    min = float('inf')
    for i in combinations_vp:
        vp_u , vp_v = i
        f, K = intrinsic_parameters(img, vp_u, vp_v)
        rotation_matrix = rotation(vp_u, vp_v, f)
##perimeter_lines ---- first_horizontal, second_horizontal, first_vertical, second_vertical------([slop, intercept],...)
# i want to add point vp_u to the first_horizontal, second_horizontal, and vpv to first_vertical, second_vertical
        np.append(perimeter_lines[0], vp_u)
        np.append(perimeter_lines[1], vp_u)
        np.append(perimeter_lines[2], vp_v)
        np.append(perimeter_lines[3], vp_v)
        fitted_lines_parameters = [line_fit_ransac(img, line) for line in perimeter_lines]
        A_point_im = intersection_points_detect(fitted_lines_parameters[0], fitted_lines_parameters[3])
        D_point_im = intersection_points_detect(fitted_lines_parameters[0], fitted_lines_parameters[2])
        dist = translation(vp_u, f, img, rotation_matrix, all_corners, square_size_big, pattern_size, A_point_im, D_point_im  )
        if dist <min:
            min = dist
    print(min)
    plt.show(block=True)