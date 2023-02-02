import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import read_pattern, read, save
from pathlib import Path
from baygeohasher import BayGeoHasher
from scipy.spatial.transform import Rotation as R
import pickle


def add_noise(points, std):
    """
    Rotates points situated on the unit sphere by a random rotation whose angle
    is sampled from a normal distribution

    :param points np.array [n x 3]: Array of points to be "polluted"
    :param std float: standard deviation of the normal distribution
    """
    n_points = points.shape[0]
    polluted_points = []
    rot_axes = np.random.uniform(0, 1, (n_points, 3))
    # normalize the Rotation axes
    rot_axes = np.divide(
        rot_axes, np.linalg.norm(rot_axes, axis=1).reshape((n_points, 1))
    )
    rot_ang = np.random.normal(0, std, (n_points, 1))
    rotvecs = np.multiply(rot_ang, rot_axes)
    for i in range(n_points):
        rot = R.from_rotvec(rotvecs[i])
        polluted_points.append(rot.apply(points[i]))
    return np.array(polluted_points)


def apply_random_rot(points):
    rot_vec = np.random.uniform(0, np.pi, 3)
    rot = R.from_rotvec(rot_vec)
    return rot.apply(points)


def get_random_view(ref_pattern, n_points, std):
    """
    Generates a random view of the ref_pattern. It returns n_points observable points for
    a random rotation with noise added to the points' position.

    :param ref_pattern np.array: Reference dot pattern [n x 3]
    :param n_points int: Number of points to be returned
    :param std float: Standard deviation of the normal distribution used to pollute the points' position [rad]
    """
    points = apply_random_rot(ref_pattern)
    points = add_noise(points, std)
    # Only keep the points that are observable. We assume that we are looking from
    # the top
    valid_points = np.argwhere(points[:, 2] > 1 / 3).flatten()
    if len(valid_points) < n_points:
        return (
            None,
            None,
        )
    valid_points = np.random.choice(valid_points, n_points, replace=False)
    pattern = points[valid_points]
    return pattern, valid_points


if __name__ == "__main__":
    # Settings of the MC benchmark test
    n_samples = 500
    n_points_used = 4
    stds = np.linspace(0, 0.1, 10)
    prj_scales = np.logspace(-2, 0.5, 6)

    # Initialize the Bayesian Geometric Hasher
    pattern_path = Path("../data/ref_pattern.csv")
    # pattern_path = Path("points_opt.csv")
    ref_pattern = read_pattern(pattern_path)
    geohasher = BayGeoHasher()
    geohasher.gen_hash_table(ref_pattern)

    log_dicts = []  # dict to log all the metric

    for prj_scale in prj_scales:
        print("Testing projection scale: {}".format(prj_scale))
        geohasher.projection_ll_scale = prj_scale
        ratios = []
        for std in tqdm(stds):
            print("For std = {}".format(std))
            success = 0
            not_enough_points = 0
            for i in range(n_samples):
                points, gt_idx = get_random_view(ref_pattern, n_points_used, std)
                if points is None:
                    not_enough_points += 1
                    continue
                idx, rot, rmse = geohasher.identify(points)
                if (gt_idx[:2] == idx).all():
                    success += 1

            print(
                "{} % cases where not enough points".format(
                    100 * not_enough_points / n_samples
                )
            )
            ratio = success / n_samples
            print("{} % success rate".format(100 * ratio))
            print(
                "{} % success rate (if ignoring when there are not enough observable points)".format(
                    100 * success / (n_samples - not_enough_points)
                )
            )
            ratios.append(ratio)

        log_dicts.append({"prj_scale": prj_scale, "ratios": ratios, "stds": stds})
        plt.plot(stds, ratios, label=prj_scale)

    # Save the test results for latter use
    save(log_dicts, "old_pattern_geohashing_MC.pkl")

    # log = read("geohashing_MC.pkl")
    plt.title(
        "Geohashing sensitivity to inaccurate dot positions (MC with {})".format(
            n_samples
        )
    )
    plt.ylabel("Identification success rate")
    plt.xlabel("Std of the noise by which each dot was rotated [rad]")
    plt.grid(True)
    plt.legend(loc="lower left", title="Alpha")
    plt.show()
