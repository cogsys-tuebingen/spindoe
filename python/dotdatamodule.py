"""
Dot dataset pytorch ligntning datamodule
It handles all the data augmentation necessary for training.
"""

import pytorch_lightning as pl
from torch.distributions import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from pathlib import Path
from typing import Optional
from torchvision import transforms as T
import re
import numpy as np
from dotdataset import DotDataset
from data_augmentation import MotionBlur, RandomFlip, CustomCompose


data_aug_transform = CustomCompose(
    [
        # MotionBlur(),
        RandomFlip(),
        # T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05), # TODO: Add color jitter for two elements
    ]
)


class DotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        RGB: bool = True,
        num_workers: int = 8,
        data_aug: bool = False,  # TODO: Implement data_auf
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.RGB = RGB
        self.data_aug = data_aug
        if self.data_aug:
            self.transform = data_aug_transform
        else:
            self.transform = None

    def setup(self, stage: Optional[str] = None):
        # Get all the index files
        self.index_paths = list(self.data_dir.glob("**/index.csv"))
        # Seperate the different datasets into train, val, test according to their rotation axis defined by their dot
        rot_axis = []
        for path in self.index_paths:
            # print(str(path))
            result = re.search("ball_dot_(.*)_up_to", str(path))
            dot_idx = int(result.group(1))
            rot_axis.append(dot_idx)
        ax_bin_count = np.bincount(rot_axis)
        arg_sorted_axis = np.argsort(ax_bin_count)
        self.datasets = []
        i = 0
        while ax_bin_count[arg_sorted_axis[i]] == 0:
            i += 1
        self.ax_test = [arg_sorted_axis[i]]
        i += 1
        self.ax_val = [arg_sorted_axis[i]]
        self.ax_train = arg_sorted_axis[i + 1 :]
        # print(self.ax_train)
        # print(self.ax_val)
        # print(self.ax_test)
        self.train_datasets = []
        self.test_datasets = []
        self.val_datasets = []
        for path in self.index_paths:
            result = re.search("ball_dot_(.*)_up_to", str(path))
            dot_idx = int(result.group(1))
            # Train dataset
            if dot_idx in self.ax_train:
                self.train_datasets.append(
                    DotDataset(path, RGB=self.RGB, transform=self.transform)
                )
            # Tets datasaet
            elif dot_idx in self.ax_test:
                self.test_datasets.append(
                    DotDataset(path, RGB=self.RGB, transform=self.transform)
                )
            # Validation dataset
            elif dot_idx in self.ax_val:
                self.val_datasets.append(
                    DotDataset(path, RGB=self.RGB, transform=self.transform)
                )
        self.data_train = ConcatDataset(self.train_datasets)
        self.data_test = ConcatDataset(self.test_datasets)
        self.data_val = ConcatDataset(self.val_datasets)
        self.n_train = len(self.data_train)
        self.n_test = len(self.data_test)
        self.n_val = len(self.data_val)
        self.n_imgs = self.n_train + self.n_test + self.n_val

    def train_dataloader(self):
        return DataLoader(
            self.data_train, self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.data_test, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    data_dir = Path(
        "/home/gossard/Code/tt_ws/src/tt_tracking_system/tt_spindetection/spin_motor_dots_andro_ball/"
    )
    data_module = DotDataModule(data_dir)
    data_module.setup()
    print("Number of train data: {}".format(data_module.n_train))
    print("Number of val data: {}".format(data_module.n_val))
    print("Number of test data: {}".format(data_module.n_test))
