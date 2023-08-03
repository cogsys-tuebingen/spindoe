"""
SpinDOE class for the spin estimation from images 
"""
import time
import typing
from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from doe import DOE
from scipy.spatial.transform import Rotation as R
from spin_regressor import SpinRegressor
from utils import get_time


class SpinDOE:
    def __init__(self, dot_detector_model):
        self.doe = DOE(dot_detector_model, True)
        self.spin_regressor = SpinRegressor()

    def estimate(self, t, imgs):
        assert len(t) == len(imgs)
        rots = []
        heatmaps = []
        filt_t = []
        filt_rots = []
        spin_vecs = []
        valid_idx = []
        for i in range(len(imgs)):
            rot, heatmap, mask, keypoints = self.doe.estimate_single(imgs[i])
            # Filter out the captures where the ball orientation is not gotten
            if rot is not None:
                filt_rots.append(rot)
                valid_idx.append(i)
                filt_t.append(t[i])
            rots.append(rot)
            heatmaps.append(heatmap)

        valid_idx = np.array(valid_idx)
        filt_t = np.array(filt_t)
        if len(valid_idx) < 5:
            return None, rots, heatmaps, valid_idx
        spin, spin_valid_idx = self.spin_regressor.RANSAC_regress(filt_t, filt_rots)

        valid_idx = valid_idx[spin_valid_idx]

        return spin, rots, heatmaps, valid_idx

    def predicted_rots(self, init_rot, t0, ts, spin_vec):
        dt = ts - t0
        spin_vecs = spin_vec.reshape(3, 1) * dt.reshape(1, -1)
        spin_rots = R.from_rotvec(spin_vecs.T)
        pred_rots = spin_rots * init_rot
        return pred_rots

    def plot_dots(self, img, rot, color=None):
        img = self.doe.reproject_dots(rot, img, color)
        return img

    def debug(self, t, imgs):
        assert len(t) == len(imgs)
        spin, rots, heatmaps, valid_idx = self.estimate(t, imgs)
        # spin = -spin
        if spin is not None:
            pred_rots = self.predicted_rots(
                rots[valid_idx[0]], t[valid_idx[0]], t, spin
            )
        n = len(imgs)
        aug_imgs = []
        for i in range(n):
            # Plot the rotations estimated with DOE
            if rots[i] is None:
                aug_img = imgs[i]
            else:
                aug_img = self.plot_dots(imgs[i], rots[i])
            if spin is not None:
                aug_img = self.plot_dots(aug_img, pred_rots[i], (255, 255, 255))
            # fig, axs = plt.subplots(3)
            # axs[0].imshow(heatmap)
            # axs[1].imshow(mask)
            # axs[2].imshow(aug_img)
            # plt.show()
            # print(rot)
            aug_imgs.append(aug_img)

        fig, axs = plt.subplots(4, 8)
        for i in range(np.min([n, 16])):
            axs[2 * (i // 8), (i % 8)].imshow(aug_imgs[i])
            axs[2 * (i // 8), (i % 8)].set_title(t[i], fontdict={"fontsize": 10})
            axs[2 * (i // 8) + 1, (i % 8)].imshow(heatmaps[i])
            if spin is not None:
                if i in valid_idx:
                    self.valid_ax(axs[2 * (i // 8), (i % 8)])
                else:
                    self.invalid_ax(axs[2 * (i // 8), (i % 8)])
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        return spin, rots, heatmaps, valid_idx

    def valid_ax(self, ax):
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2.5)  # change width
            ax.spines[axis].set_color("green")

    def invalid_ax(self, ax):
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2.5)  # change width
            ax.spines[axis].set_color("red")


if __name__ == "__main__":

    dot_detector_model = Path().cwd().parent / "data" / "model" / "spindoe.ckpt"
    spindoe = SpinDOE(dot_detector_model)
    parser = ArgumentParser(
        prog="SpinDOE", description="Estimates the spin of a dotted table tennis ball"
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=Path(
            "../data/test/",
            help="Directory where the sequential ball images are saved",
        ),
    )
    args = parser.parse_args()
    # Get the images from the test directory
    img_paths = sorted(list(args.dir.glob("*.png")))
    imgs = []
    times = []
    for path in img_paths:
        t = get_time(path)
        times.append(t)
        img = cv2.imread(str(path))
        img = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # Because probably read with cv2.imread
        imgs.append(img)

    times = np.array(times)
    spin, rots, heatmaps, valid_idx = spindoe.debug(times, imgs)

    # IMPORTANT
    # Spin is in the frame of the first valid orientation. So it needs to be
    # transformed back to the frame used for the reference dot pattern.
    corrected_spin = rots[valid_idx[0]].inv().apply(spin)
    print("Spin in rad/s", corrected_spin)
    print("spin norm in rps", np.linalg.norm(corrected_spin) / (2 * np.pi))
