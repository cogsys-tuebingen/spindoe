"""
Dotted-ball Orientation Estimator (DOE)
"""

from dotdetector import DotDetector
import time
from pathlib import Path
from baygeohasher import BayGeoHasher
import matplotlib.pyplot as plt
from dotdataset import read_ball_images
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms as T
from utils import read_pattern
import cv2
import threading
from queue import Queue

NUM_THREADS = 10
q = Queue()


class DOE:
    def __init__(self, dot_detector_model_path, use_gpu=False) -> None:
        self.use_gpu = use_gpu
        self.thresh_ratio = 0.7
        self.size = 60  # Size of the image input
        self.rmse_thres = 100  # rmse theshold for estimated rotation to be accepted

        # Dots' Color for GUI
        self.dot_color = (255, 0, 0)
        self.logo_color = (0, 255, 0)

        # Initialize the different components of DOE
        self.init_blob_detector()
        self.init_dot_detector(dot_detector_model_path)
        self.init_geohasher()
        print("DOE  initiated")

    def init_dot_detector(self, model_path):
        self.dot_detector = DotDetector.load_from_checkpoint(str(model_path))
        if self.use_gpu:
            self.dot_detector.eval().cuda(device=0)
        else:
            self.dot_detector.eval()

    def init_geohasher(self):
        self.ref_pattern = read_pattern("../data/ref_pattern.csv")
        self.geohasher = BayGeoHasher()
        self.geohasher.gen_hash_table(self.ref_pattern)

    def init_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = False
        # params.minCircularity = 0.2
        # params.maxCircularity = 0.2
        params.filterByConvexity = False
        # params.minConvexity = 0.9
        # params.maxConvexity = 0.9
        params.filterByInertia = False
        # params.minInertiaRatio = 0
        # params.maxInertiaRatio = 0
        params.filterByArea = True
        params.minArea = 1
        params.maxArea = 30
        params.filterByColor = True
        params.blobColor = 255
        params.minDistBetweenBlobs = 1
        # params.minThreshold = 100
        # params.maxThreshold = 255
        # params.thresholdStep = 20
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

    def extract_dots(self, heatmap):
        mask = cv2.inRange(heatmap, self.thresh_ratio, (255))
        # mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        # plt.imshow(mask)
        # plt.show()
        keypoints = self.blob_detector.detect(mask)
        # print("number of keypoints detected: {}".format(len(keypoints)))
        return mask, keypoints

    def estimate_single(self, img):
        tensor = self.process_input(img)
        # print(tensor)
        heatmap = self.dot_detector(tensor[None, ...])[0]
        if self.use_gpu:
            heatmap = heatmap.cpu().detach().numpy().squeeze()
        else:
            heatmap = heatmap.detach().numpy().squeeze()

        # plt.imshow(heatmap)
        # plt.show()
        mask, keypoints = self.extract_dots(heatmap)
        # DOE cannot work with less than 3 points
        if len(keypoints) < 3:
            return None, mask, heatmap

        dots = self.kps2pt3d(keypoints)

        idx, rot, rmse = self.geohasher.identify(dots)

        # If the "reprojection error" is too big, doe is assumed to have failed
        if rmse > self.rmse_thres:
            print("RMSE too high")
            return None, mask, heatmap

        return rot, mask, heatmap

    def estimate_multi(self, imgs):
        concat_tensor = []
        for img in imgs:
            tensor = self.process_input(img)
            concat_tensor.append(tensor)
        concat_tensor = torch.stack(concat_tensor)
        # print(tensor)
        heatmaps = self.dot_detector(concat_tensor)
        if self.use_gpu:
            heatmaps = heatmaps.cpu().detach().numpy().squeeze()
        else:
            heatmaps = heatmaps.detach().numpy().squeeze()

        rots = []
        masks = []
        for heatmap in heatmaps:
            # plt.imshow(heatmap)
            # plt.show()
            mask, keypoints = self.extract_dots(heatmap)
            # DOE cannot work with less than 3 points
            if len(keypoints) < 3:
                return None, mask, heatmap

            dots = self.kps2pt3d(keypoints)

            idx, rot, rmse = self.geohasher.identify(dots)

            # If the "reprojection error" is too big, doe is assumed to have failed
            if rmse > self.rmse_thres:
                rots.append(None)
            else:
                rots.append(rot)
            masks.append(mask)

        return rots, masks, heatmaps

    def multi_treaded_estimate(self, imgs):
        concat_tensor = []
        for img in imgs:
            tensor = self.process_input(img)
            concat_tensor.append(tensor)
        concat_tensor = torch.stack(concat_tensor)
        # print(tensor)
        heatmaps = self.dot_detector(concat_tensor)
        if self.use_gpu:
            heatmaps = heatmaps.cpu().detach().numpy().squeeze()
        else:
            heatmaps = heatmaps.detach().numpy().squeeze()

        rots = []
        masks = []
        for heatmap in heatmaps:
            q.put(heatmap)
        for t in range(NUM_THREADS):
            worker = threading.Thread(target=self.dot_2_rots)
            worker.daemon = True
            worker.start()
        q.join()

        return None, None, None

    def dot_2_rots(self):
        global q
        while True:
            heatmap = q.get()
            mask, keypoints = self.extract_dots(heatmap)
            # DOE cannot work with less than 3 points
            if len(keypoints) < 3:
                return None, mask, heatmap

            dots = self.kps2pt3d(keypoints)

            idx, rot, rmse = self.geohasher.identify(dots)
            q.task_done()

    def kps2pt3d(self, kps):
        n = len(kps)
        pts3d = np.zeros((n, 3))
        for i in range(n):
            x = 2 * (kps[i].pt[0] / self.size - 1 / 2)
            y = -2 * (kps[i].pt[1] / self.size - 1 / 2)
            if x**2 + y**2 > 1:
                z = 0
            else:
                z = np.sqrt(1 - x**2 - y**2)
            pts3d[i, :] = np.array([x, y, z])
        return pts3d

    def process_input(self, img):
        # Check the type of the input and the range of values
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                tensor = T.ToTensor()(img)
            else:
                img = cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB
                )  # Because probably read with cv2.imread
                # plt.imshow(img)
                # plt.show()
                tensor = T.ToTensor()(img)

        # elif type(img):
        #     pass
        else:
            raise TypeError("DOE input type not supported")
        if self.use_gpu:
            tensor = tensor.cuda(device=0)
        tensor = T.Resize((self.size, self.size))(tensor)
        return tensor

    def reproject_dots(self, rot, img, color=None):
        points_rotated = rot.inv().apply(self.ref_pattern)
        logo_rotated = rot.inv().apply(np.array([[0, 0, 1]]))
        if color is None:
            pattern_img = self.draw_visible_3dpoints_on_image(
                points_rotated, img, self.dot_color
            )
            pattern_img = self.draw_visible_3dpoints_on_image(
                logo_rotated, pattern_img, self.logo_color
            )
        else:
            pattern_img = self.draw_visible_3dpoints_on_image(
                points_rotated, img, color
            )
        return pattern_img

    def draw_visible_3dpoints_on_image(self, points, img, color, alpha=1):
        visible_points = [pt for pt in points if pt[2] > 0]
        height, width = img.shape[:2]
        res_img = img.copy()
        for pt in visible_points:
            x = int((pt[0] + 1) * 0.5 * width)
            y = int((-pt[1] + 1) * 0.5 * height)
            res_img = cv2.circle(res_img, (x, y), radius=1, color=color, thickness=-1)
        return res_img


if __name__ == "__main__":
    use_gpu = True
    dot_detector_model = Path(
        "/home/gossard/Git/spindoe/python/lightning_logs/version_27/checkpoints/epoch=4-step=1100.ckpt"
    )
    # Test the doe
    doe = DOE(dot_detector_model, use_gpu)
    # Get the images from the test directory
    img_dir = Path.cwd().parent / "data" / "test"
    img_paths = list(img_dir.glob("*.png"))

    # Process
    # rdm_idx = np.random.choice(np.arange(len(img_paths)), 6, replace=False)
    rdm_idx = np.arange(10)
    aug_imgs = []
    heatmaps = []
    t = []
    for i in range(10):
        imgs = []
        for idx in rdm_idx:
            img = cv2.imread(str(img_paths[idx]))
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        rots, masks, heatmaps = doe.estimate_multi(imgs)
    for i in range(1000):
        imgs = []
        for idx in rdm_idx:
            img = cv2.imread(str(img_paths[idx]))
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        t1 = time.time()
        rots, masks, heatmaps = doe.estimate_multi(imgs)
        t.append(time.time() - t1)
        # rot, mask, heatmap = doe.estimate(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # aug_img = doe.reproject_dots(rot, img)
        # fig, axs = plt.subplots(3)
        # axs[0].imshow(heatmap)
        # axs[1].imshow(mask)
        # axs[2].imshow(aug_img)
        # plt.show()
        # print(rot)
        # heatmaps.append(heatmap)
        # aug_imgs.append(aug_img)

    t = np.array(t)
    print(np.mean(t))
    print(np.std(t))
    # print("Runtime: {}".format(time.time() - t1))
    fig, axs = plt.subplots(2, 6)
    for i in range(6):
        axs[i % 2, 2 * (i % 3)].imshow(aug_imgs[i])
        axs[i % 2, 2 * (i % 3) + 1].imshow(heatmaps[i])
    plt.show()