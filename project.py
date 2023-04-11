import cv2
import os
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
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
    #corners = cv2.cornerSubPix(gray_scale, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(gray_scale, pattern_size, corners, ret)
    # if ret == True: # if the detection was succesful or not
#@Draw and display the corners
        # cv2.drawChessboardCorners(gray_scale, pattern_size, corners, ret)
        # cv2.imshow("", gray_scale)
        # cv2.waitKey(0)

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


def line_fit_ransac(img, pattern_size, all_corners):
    lines = \
        parimeter_lines_detect(img, pattern_size, all_corners)

    parameters = []

    height, width, _ = img.shape
    for line in lines:
        X = line[:, 0].reshape(-1, 1)
        y = line[:, 1].reshape(-1, 1)
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        # The coefficients of the fitted line
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        parameters.append([slope, intercept])
        #draw the lines on top of the image
        x1 = 0
        y1 = int(slope * x1 + intercept)
        x2 = width - 1
        y2 = int(slope * x2 + intercept)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # # Display the image
    # cv2.imshow('Fitted line', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    parameters = np.squeeze(np.array(parameters))
    return np.array(parameters)


def intersection_points_detect(parameters):
    points = []
    for i in range(0, 4, 2):
        slope1, intercept1 = parameters[i]
        slope2, intercept2 = parameters[i + 1]
        # Solve for the x and y values where the two lines intersect
        x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
        y_intersect = slope1 * x_intersect + intercept1
        # Print the intersection point
        # print("Vanishing point: ({}, {})".format(x_intersect, y_intersect))
        points.append((x_intersect, y_intersect))

        cv2.circle(img, (int(x_intersect), int(y_intersect)), 5, (255, 0, 0), -1)

    # cv2.imshow('Chessboard Corners', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return points

def distance(point1, point2):# Calculates the distance between two points in a 2D coordinate system.
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    # print("point1", point1, "x1", x1, "y1", y1)
    # print("point2", point2, "x2", x2, "y2", y2)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def focal_length_calc(vp1,vp2,p, p_uv):
    OcVi_len = math.sqrt(distance(vp1, p_uv ) * distance(p_uv , vp2))
    focal_length =math.sqrt( pow(OcVi_len, 2) -pow(distance(p, p_uv), 2))
    #print("focal_length",focal_length)
    return focal_length

def principal_point_coordinates(img):
    # Get the dimensions of the image
    height, width, channels = img.shape
    # Calculate the center point
    center_x = int(width / 2)
    center_y = int(height / 2)
    # Print the center point
    #print("Center point: ({}, {})".format(center_x, center_y))
    return [center_x, center_y]

def intrinsic_parameters(img, vp1, vp2):
    p_point = principal_point_coordinates(img)  # u0, v0 for the intrinsic matrix
    u0= p_point[0]
    v0= p_point[1]
    skew = 0
    scale_factor_u = scale_factor_v = 1
    # Convert points to numpy arrays
    vp1 = np.array(vp1)
    vp2 = np.array(vp2)
    p = np.array(p_point)

    # Calculate the unit vector of P1P2
    vp1vp2 = vp2 - vp1
    vp1vp2_unit = vp1vp2 / np.linalg.norm(vp1vp2)
    # Calculate the vector projection of OP onto P1P2
    pvp1 = p - vp1
    op_proj = np.dot(pvp1, vp1vp2_unit) * vp1vp2_unit
    # Calculate the coordinates of point P
    # Puv be the orthogonal projection of P on the the line (VP1 VP2)
    p_uv = vp1 + op_proj

    f=focal_length_calc(vp1,vp2,p, p_uv) # focal length
    K =np.array([[scale_factor_u*f , skew,            u0],
        [0,                 scale_factor_v*f, v0],
        [0,                   0,               1]])
    # print("intrinsic matrix", K)
    return f, K


def rotation(intersection_points, f):
    vp1 = intersection_points[0]
    vp2 = intersection_points[1]

    len_u_rc = math.sqrt(vp1[0]**2 + vp1[1]**2+ f**2)
    len_v_rc = math.sqrt(vp2[0] ** 2 + vp2[1] ** 2 + f ** 2)
    v_rc = np.array([vp1[0],vp1[1], f]) / len_v_rc
    u_rc = np.array([vp2[0], vp2[1], f]) / len_u_rc
    w_rc =np.cross(u_rc, v_rc)

    v_rc = v_rc.reshape(3,1)
    u_rc= u_rc.reshape(3,1)
    w_rc= w_rc.reshape(3,1)

    # rotation from world to camera
    rotation = np.hstack((u_rc, v_rc, w_rc))
    #print("rotation", rotation)
    return rotation

# point1 and point2 are coordinates of the side of the pattern in world coordinates
#  we are setting point P in the real world  = length of 4 squares of the pattern
# look for the p' point in the image (p_im)- find as 5th point on the first horizontal line


def construct_plane(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    d = -np.dot(normal, np.array(p1))
    return np.concatenate((normal, [d])), normal

def translation(f, img, r, point1, point2, all_corners, square_size, pattern_size ):
    x, y = pattern_size
    principal_point = principal_point_coordinates(img)  # image coordinates of the principal point

    k_im = all_corners[int(x/2 -1)] # coordinates of k' in the pixel coordinates, k' lies on the projection line a'd'
    # define the a' and d' points - projections of the longest side of the pattern to the image plane
    a_im = all_corners[int(x - 1)]
    d_im =all_corners[0]
    k_u, k_v = k_im # x, y in pixels
    # x*1.8*2.54/ 3.07
    # y*1* 2.54/ 2.048
    p_u, p_v = principal_point
    a_u, a_v = a_im
    k_rc = np.array([ (k_u - p_u)*1.8*2.54/ 3.07, (k_v - p_v)*1* 2.54/ 2.048, f])
    a_rc = np.array([ (a_u - p_u)*1.8*2.54/ 3.07,(a_v - p_v)*1* 2.54/ 2.048, f])
    o_rc = np.zeros(3)

    #  each square is 125 mm for the A0 pattern
    l = 4* square_size
    AK_ro = l * np.array([1,0,0]).reshape(3,1)  # l * u where u is a unit vector of the world coordinate system
    print("AK_ro", AK_ro)
    AK_rc =np.dot(r,AK_ro).reshape(3,) #AP_rc = M_oc*AP_ro ----d line
    print("AK_rc", AK_rc)
    k_new = -10*AK_rc+a_rc
    print("k new", k_new)
    AK_rc_unit= AK_rc / np.linalg.norm(AK_rc) # unit vector
    AK_rc_from_a = a_rc -100*AK_rc_unit

    print("a_rc",  a_rc)
    print("k_rc",  k_rc)
    print("o_rc", o_rc)
    AKO_rc, n  = construct_plane(a_rc, k_rc, o_rc)
    print("plane equation", AKO_rc)
    # Normalize the normal vector (optional)
    normal_from_o = o_rc + n
    plane_norm_unit = n / np.linalg.norm(n)
    print("normal unit", plane_norm_unit)

    # AP_proj = project_onto_plane(AK_rc, normal_unit, o_rc)
    # p_intersection_KO = intersection_lines(AP_proj, o_rc, k_rc)
    # print("p_intersection_KO", p_intersection_KO)

    distance_plane_to_vector = np.abs(AK_rc.dot(plane_norm_unit))
    print("distance_plane_to_vector", distance_plane_to_vector)
    # Find the dot product between the vector and normal vector of the plane
    dot_prod = np.dot(AK_rc_from_a, plane_norm_unit)
    # Calculate the magnitudes of the vector and normal vector of the plane
    vec_mag = np.linalg.norm(AK_rc_from_a)
    normal_mag = np.linalg.norm(plane_norm_unit)
    # Calculate the angle between the vector and plane
    angle = np.arccos(dot_prod / (vec_mag * normal_mag))
    print("angle", angle)


#VISUALIZATION
    # plotting a surface
    DATA = np.array([k_rc,a_rc, o_rc])
    Xs = DATA[:, 0]
    Ys = DATA[:, 1]
    Zs = DATA[:, 2]
    #Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
    # ax.quiver(o_rc[0], o_rc[1], o_rc[2], n[0], n[1], n[2], color='b')  # ok_r
    # ax.quiver(o_rc[0], o_rc[1], o_rc[2], normal_from_o[0], normal_from_o[1], normal_from_o[2], color='b')  # ok_rc

    #Plot the vector as an arrow
    ax.quiver(a_rc[0], a_rc[1], a_rc[2], AK_rc[0], AK_rc[1], AK_rc[2], color='r') # ok_rc

    #Plot the points as spheres
    ax.scatter(a_rc[0], a_rc[1], a_rc[2], color='b', marker='o')
    ax.scatter(k_rc[0], k_rc[1], k_rc[2], color='b', marker='o')
    ax.scatter(o_rc[0], o_rc[1], o_rc[2], color='b', marker='o')
    ax.scatter(k_new[0], k_new[1], k_new[2], color='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return None

# def project_onto_plane(v, normal,o):
#     # Compute the dot product of the normal vector with the vector you want to project.
#     signed_distance = np.dot(normal, v - o)
#     # Subtract the projection of the vector onto the normal vector from the original vector-  Calculate the projection of vector AP onto the plane OAK.
#     projection = v - signed_distance / np.dot(normal, normal) * normal
#     return projection

def intersection_lines(projection, o_rc, k_rc):
    # Find the equation of the line KO.
    # The line passes through K and O, so we can write it as K + t*(O-K).
    # Here, we define the direction vector d = O - K.
    d = o_rc - k_rc
    #  Find the point of intersection P between the line KO and the projected vector from step 2.
    # We substitute the equation of the line into the equation for the projected vector, and solve for t.
    t = np.dot(projection - k_rc, d) / np.dot(d, d)
    # Finally, we calculate the point of intersection P as K + t*(O-K).
    P = k_rc + t*d
    # The point P is the intersection of the line KO and the projection of vector AP onto the plane OAK.
    return P

def point_on_line_segment_3d(p1, p2, test_point):
    d = abs(np.cross((p2-p1), (p1-test_point))) / abs(p2-p1)
    print("d", d)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('MacOSX')
    filelist = list(Path('.').glob('**/*.mp4'))
    pattern_size = (8, 5)
    square_size_big = 125
    img, all_corners = pattern_corner_detect(pattern_size)
    parimeter_lines_detect(img, pattern_size, all_corners)
    fitted_lines_parameters = line_fit_ransac(img, pattern_size, all_corners)  # [[slop, intercept],...]
    intersection_points = intersection_points_detect(fitted_lines_parameters)
    f, K = intrinsic_parameters(img, intersection_points[0], intersection_points[1])
    rotation_matrix = rotation(intersection_points, f)
    t = translation(f, img, rotation_matrix,intersection_points[0], intersection_points[1], all_corners, square_size_big, pattern_size  )
    plt.show(block=True)