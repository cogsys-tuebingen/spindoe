"""
SpinDOE class for the spin estimation from images 
"""
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import numpy as np
import typing
from doe import DOE
from utils import get_time
import time
from spin_regressor import SpinRegressor


class SpinDOE:
    def __init__(self, dot_detector_model):
        self.doe = DOE(dot_detector_model, True)
        self.spin_regressor = SpinRegressor()

    def estimate(self, t, imgs):
        rots = []
        heatmaps = []
        filt_t = []
        filt_rots = []
        for i in range(len(imgs)):
            # img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
            rot, mask, heatmap = self.doe.estimate_single(imgs[i])
            # plt.imshow(heatmap)
            # plt.show()
            if rot is not None:
                filt_rots.append(rot)
                filt_t.append(t[i])
            rots.append(rot)
            heatmaps.append(heatmap)

        filt_t = np.array(filt_t)
        spin, valid_idx = self.spin_regressor.RANSAC_regress(filt_t, filt_rots)

        return spin, rots, heatmaps, valid_idx

    def debug(self, imgs, rots, heatmaps, valid):
        assert len(imgs) == len(rots)
        n = len(imgs)
        aug_imgs = []
        for i in range(n):
            img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
            if rots[i] is None:
                aug_img = img
            else:
                aug_img = self.doe.reproject_dots(rots[i], img)
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
            axs[2 * (i // 8) + 1, (i % 8)].imshow(heatmaps[i])
            if i in valid:
                self.valid_ax(axs[2 * (i // 8), (i % 8)])
            else:
                self.invalid_ax(axs[2 * (i // 8), (i % 8)])
        plt.show()

    def valid_ax(self, ax):
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2.5)  # change width
            ax.spines[axis].set_color("green")

    def invalid_ax(self, ax):
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2.5)  # change width
            ax.spines[axis].set_color("red")


if __name__ == "__main__":

    dot_detector_model = Path(
        "/home/gossard/Git/spindoe/python/lightning_logs/version_41/checkpoints/epoch=5-step=1320.ckpt"
    )
    # Get the images from the test directory
    # img_dir = Path.cwd().parent / "data" / "test"
    img_dir = Path("/home/gossard/Nextcloud/tabletennis/trajectory_dataset/9")
    img_paths = sorted(list(img_dir.glob("*.png")))
    imgs = []
    times = []
    for path in img_paths:
        t = get_time(path)
        times.append(t)
        img = cv2.imread(str(path))
        img = img[11:-11, 11:-11]
        # plt.imshow(img)
        # plt.show()
        imgs.append(img)

    times = np.array(times)
    # parser = ArgumentParser(
    #     prog="SpinDOE", description="Estimates the spin of a dotted table tennis ball"
    # )
    # parser.add_argument(
    #     "-d",
    #     "--dir",
    #     default=Path(
    #         "../data/test/", help="Directory where the sequential ball images are saved"
    #     ),
    # )
    # args = parser.parse_args()
    spindoe = SpinDOE(dot_detector_model)
    t1 = time.time()
    spin, rots, heatmaps, valid_idx = spindoe.estimate(times, imgs)
    print("Runtime: {}".format(time.time() - t1))
    # print(rots)
    spindoe.debug(imgs, rots, heatmaps, valid_idx)

    print(spin)
