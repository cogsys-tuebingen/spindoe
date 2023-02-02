"""
Script to test the dot detector on the images in the data/test directory
"""

from dotdetector import DotDetector
from pathlib import Path
from dotdataset import read_ball_images
import matplotlib.pyplot as plt
import time
import torch
from torchvision import transforms as T
import numpy as np

if __name__ == "__main__":
    torch.device("cuda")
    # torch.device("cpu")
    # Get the images from the test directory
    img_dir = Path.cwd().parent / "data" / "test"
    img_paths = list(img_dir.glob("*.png"))

    dot_detector_model = Path(
        "/home/gossard/Git/spindoe/python/lightning_logs/version_16/checkpoints/epoch=28-step=6380.ckpt"
    )
    dot_detector = DotDetector.load_from_checkpoint(str(dot_detector_model))
    # dot_detector.eval().cuda(device=0)
    dot_detector.eval()

    # Process
    rdm_idx = np.random.randint(0, len(img_paths), 3)
    imgs = read_ball_images([img_paths[i] for i in rdm_idx])
    # imgs = imgs.cuda(device=0)

    t1 = time.time()
    y_hat = dot_detector(imgs)
    print("Runtime: {}".format(time.time() - t1))
    # print(y_hat.size())
    fig, axs = plt.subplots(2, 3)
    for i in range(3):
        heatmap = T.ToPILImage()(y_hat[i])
        img = T.ToPILImage()(imgs[i])
        axs[0, i].imshow(img)
        axs[1, i].imshow(heatmap)
    plt.show()
