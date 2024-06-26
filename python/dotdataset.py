"""
Dataset class for the dotted ball images
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image


def read_ball_image(img_path, RGB=True):
    img = read_image(str(img_path))
    if not RGB:
        img = T.Grayscale()(img)
    # Convert image to range [0,1] and floats
    img = img.float() / 255.0
    return img


def read_ball_images(img_paths, size=60, RGB=True):
    imgs = []
    for path in img_paths:
        img = read_ball_image(path, size=size, RGB=RGB)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


class DotDataset(Dataset):
    def __init__(
        self,
        index_path,
        std_dot: float = 1.5,
        size: int = 60,
        RGB: bool = True,
        transform=None,
    ):
        super().__init__()

        # Read the index file and convert it to a panda frame
        self.index_path = index_path
        with open(self.index_path, "r") as temp_f:
            col_count = [len(l.split(",")) for l in temp_f.readlines()]
        self.max_dots = int((max(col_count) - 5) / 2)
        column_names = [i for i in range(0, max(col_count))]
        self.index = pd.read_csv(
            self.index_path, header=None, sep=",", names=column_names
        )
        self.std_dot = std_dot
        self.size = size
        self.resize = T.Resize((self.size, self.size), antialias=True)
        self.RGB = RGB
        self.transform = transform

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, idx):
        img_path = self.index[0][idx]
        quat = self.index.loc[idx, 1:4].to_numpy(dtype=np.float64)
        img = read_ball_image(img_path, RGB=self.RGB)
        orig_h, orig_w = img.shape[1:]
        img = T.Resize((self.size, self.size), antialias=True)(img)

        # Generate the heatmap
        dots = self.get_dots(idx)
        dots[:, 0] = self.size / orig_h * dots[:, 0]
        dots[:, 1] = self.size / orig_w * dots[:, 1]
        heatmap = self.gen_heatmap(dots, self.size, self.size)
        heatmap = heatmap[None, ...]

        if self.transform:
            img, heatmap = self.transform(img, heatmap)
            # plt.imshow(img.squeeze())
            # plt.show()

        return img, heatmap, quat

    def get_dots(self, idx):
        """
        Returns a array of dots's xy position [n x 2] for image idx

        :param idx int: Image index in the dataset
        """
        n_dots = 0
        dots = []
        while not (np.isnan(self.index[5 + 2 * n_dots][idx])):
            dots.append(
                [self.index[5 + 2 * n_dots][idx], self.index[6 + 2 * n_dots][idx]]
            )
            n_dots += 1
            if n_dots >= self.max_dots:
                break
        return np.array(dots)

    def gaussian_2d(self, x, y, mx, my, sx, sy):
        """2D-Normal density function
        Args:
            x (float): y evaluated
            y (float): x evaluated
            mx (float): x mean
            my (float): y mean
            sx (float): x standard deviation
            sy (float): y standard deviation

        Returns:
           probability (float):
        """
        return (
            1
            / (2 * np.pi * sx * sy)
            * torch.exp(
                -((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2))
            )
        )

    def get_img_w_dots(self, idx):
        img, heatmap, _ = self.__getitem__(idx)
        img = np.array(T.ToPILImage()(img))
        heatmap = np.array(T.ToPILImage()(heatmap))
        for dot in dots:
            x = int(dot[0])
            y = int(dot[1])
            img = cv2.circle(img, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
            heatmap = cv2.circle(
                heatmap, (x, y), radius=5, color=(255, 0, 0), thickness=1
            )
        return img, heatmap

    def gen_heatmap(self, dots, height, width):
        """Generates the heatmap for the dots given

        Args:
            dots (list Nx2): list of xy dot coordinates
            height (int): height of the original from where the dots come from
            width (int): width of the original from where the dots come from

        Returns:
            heatmap (np.array): Normalized heatmap of dot positions
        """
        heatmap = torch.zeros((self.size, self.size), dtype=torch.float)
        for dot in dots:
            x = torch.arange(self.size)
            y = torch.arange(self.size)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            single_dot_heatmap = self.gaussian_2d(
                yy, xx, dot[1], dot[0], self.std_dot, self.std_dot
            )
            single_dot_heatmap = single_dot_heatmap / torch.max(
                single_dot_heatmap
            )  # To normalize each dot map
            heatmap = torch.max(heatmap, single_dot_heatmap)
        heatmap = heatmap / torch.max(heatmap)
        return heatmap

    def show(self, idx):
        """
        Display the image and heatmap at idx
        """
        img, heatmap, _ = self.__getitem__(idx)
        img = T.ToPILImage()(img)
        heatmap = T.ToPILImage()(heatmap)

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(heatmap)
        plt.show()


if __name__ == "__main__":
    img_dir = "/home/gossard/Code/tt_ws/src/tt_tracking_system/tt_spindetection/spin_motor_dots_andro_ball/ball_dot_14_up_to/-020/index.csv"

    dataset = DotDataset(img_dir, RGB=True)
    len_data = len(dataset)
    print("This dataset contains {} images".format(len_data))

    fig, axs = plt.subplots(2, 4)
    idxs = np.random.randint(0, len_data, 4)
    for i in range(2):
        for j in range(2):
            img, heatmap = dataset.get_img_w_dots(idxs[i + j])
            axs[i, 2 * j].imshow(img)
            axs[i, 2 * j + 1].imshow(heatmap)
    plt.show()
