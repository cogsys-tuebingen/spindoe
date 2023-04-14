"""
Example script to read and visualize the trajectories
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_traj(traj_name):
    traj = np.genfromtxt(
        "trajectories/{:03d}.csv".format(int(traj_name)), delimiter=";"
    )
    return traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "idx",
        type=int,
        help="Index of the trajectory to read",
    )
    args = parser.parse_args()

    # Read the index
    index_path = Path("trajectories/index.csv")
    index = pd.read_csv(str(index_path))

    spin = index.loc[args.idx][["x_spin", "y_spin", "z_spin"]]
    print("Spin vector")
    print(spin)

    # Read the trajectory
    traj = read_traj(index.loc[args.idx]["traj_file"])

    # Plot the trajectory
    fig, axs = plt.subplots(3)
    axs[0].plot(traj[:, 0], traj[:, 1])
    axs[1].plot(traj[:, 0], traj[:, 2])
    axs[2].plot(traj[:, 0], traj[:, 3])

    axs[2].set_xlabel("Time [s]")
    axs[0].set_ylabel("X [m]")
    axs[1].set_ylabel("Y [m]")
    axs[2].set_ylabel("Z [m]")
    plt.show()
