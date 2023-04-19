from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    import matplotlib
    # matplotlib.use('MacOSX')
    matplotlib.use('QtCairo')
    filelist = list(Path('.').glob('**/*.mp4'))


