"""
Transforms used for the data augmentation of the dataset
"""
import numpy as np
from kornia.filters.motion import motion_blur
import torchvision.transforms.functional as F
from torchvision import transforms as T


# class ToTensor(object):
#     def __init__(self):
#         pass

#     def __call__(self, img, heatmap):
#         t_img = T.ToTensor()(img)
#         t_heatmap = T.ToTensor()(heatmap)
#         return t_img, t_heatmap


class CustomCompose(object):
    """
    Takes 2 arg instead of one
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tar):
        for t in self.transforms:
            img, tar = t(img, tar)
        return img, tar


class MotionBlur(object):
    def __init__(self):
        pass

    def __call__(self, img, heatmap):
        if np.random.random() > 0.2:
            kernel_size = 2 * np.random.randint(1, 6) + 1
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
