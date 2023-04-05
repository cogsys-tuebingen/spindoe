import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils
from scipy.spatial.transform import Rotation as R

r_ball = 1

file_paths = [
    "../cad/ref_pattern.csv",
]

for file_path in file_paths:
    points = utils.read_pattern(file_path)
    # points = utils.normalize_points(points)
    # rot = R.from_euler("y", 90, degrees=True)
    # points = rot.apply(points)

    ## Create display figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax.set_aspect('equal')

    ## Draw a sphere surface that represents the ball
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = r_ball * np.outer(np.cos(u), np.sin(v))
    y = r_ball * np.outer(np.sin(u), np.sin(v))
    z = r_ball * np.outer(np.ones(np.size(u)), np.cos(v))

    elev = 10.0
    rot = 80.0 / 180 * np.pi
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color="b", linewidth=0, alpha=0.2)
    ax.view_init(elev=elev, azim=0)

    ## Plot the point
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], marker=".", c="red", edgecolors="red"
    )
    for i, point in enumerate(points):
        ax.text(point[0], point[1], point[2], str(i))

    # points *= 22
    # utils.write2cvs("points_opt_n20_rotated.csv", points)
    plt.show()
