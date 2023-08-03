"""
Transforms used for the data augmentation of the dataset
"""
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from kornia.filters.motion import motion_blur
from torchvision import transforms as T


class CustomCompose(object):
    """
    Enables to compose transformations with 2 inputs (input and target)
    This is necessary for transforms like flipping
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ColorAugmentation(object):
    """
    Data augmentation for color changes
    """

    def __init__(self):
        self.color_aug = T.ColorJitter(
            brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
        )

    def __call__(self, img, heatmap):
        return self.color_aug(img), heatmap


class MotionBlur(object):
    def __init__(self):
        pass

    def __call__(self, img, heatmap):
        if np.random.random() > 0.2:
            kernel_size = 2 * np.random.randint(1, 8) + 1
            angle = 2 * np.pi * np.random.random()
            direction = -1 + 2 * np.random.random()
            img = img[None, :]
            img = motion_blur(img, kernel_size, angle, direction)
            img = img[0]
        return img, heatmap


class RandomFlip(object):
    def __init__(self, p_hflip=0.5, p_vflip=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, img, heatmap):
        # Random horizontal flipping
        if np.random.random() < self.p_hflip:
            img = F.hflip(img)
            heatmap = F.hflip(heatmap)

        # Random vertical flipping
        if np.random.random() < self.p_vflip:
            img = F.vflip(img)
            heatmap = F.vflip(heatmap)
        return img, heatmap
