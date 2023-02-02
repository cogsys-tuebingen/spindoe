import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
import pytorch_lightning as pl
from loss import FocalLoss
import numpy as np


class Layer(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn=nn.ReLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_dim, affine=False)
        self.act_fn = act_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.act_fn(x)
        x = self.batch_norm(x)
        return x


class DotDetector(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        ## Specifying the model hyperparmeters
        # Type of the input image
        if kwargs["input"] == "rgb":
            self.input = "rgb"
            prev_dim = 3
        elif kwargs["input"] == "grayscale":
            self.input = "grayscale"
            prev_dim = 1
        else:
            raise ValueError("Wrong input defined")

        # Activation function used
        if kwargs["act_fn"] == "relu":
            act_fn = nn.ReLU()
        elif kwargs["act_fn"] == "leaky_relu":
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Activation function not implemented")

        # Activation function used for the last layer
        if kwargs["last_act_fn"] == "sigmoid":
            self.last_act_fn = torch.sigmoid
        elif kwargs["last_act_fn"] == "hardsigmoid":
            self.last_act_fn = F.hardsigmoid
        else:
            raise ValueError(
                "last_act_fn: {} is not supported".format(kwargs["last_act_fn"])
            )

        # Loss used to train the model
        if kwargs["loss_fn"] == "focal_loss":
            self.loss_fn = FocalLoss()
        elif kwargs["loss_fn"] == "L1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError("loss_fn: {} not supported".format(kwargs["loss_fn"]))

        ## Creating the NN architecture
        self.layers = []
        for dim in tuple(map(int, kwargs["layers"].split(","))):
            if prev_dim not in (1, 3):
                if prev_dim < dim:
                    self.layers.append(nn.MaxPool2d(2, stride=2, padding=0))
                elif prev_dim > dim:
                    self.layers.append(nn.Upsample(scale_factor=2))
            self.layers.append(Layer(prev_dim, dim))
            prev_dim = dim
        # Add the last layer that outputs the head
        self.layers.append(nn.Conv2d(prev_dim, 1, 1, padding=0))
        self.nn = nn.Sequential(*self.layers)

        print("Saving hyperparmeters")
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CnnDotDetector")
        parser.add_argument("--last_act_fn", type=str, default="sigmoid")
        parser.add_argument("--act_fn", type=str, default="leaky_relu")
        parser.add_argument("--loss_fn", type=str, default="focal_loss")
        parser.add_argument(
            "--layers",
            type=str,
            default="64, 64, 128, 128,  256, 256, 128, 128, 64, 64",
        )
        parser.add_argument("--input", type=str, default="rgb")
        return parent_parser

    def forward(self, x):
        x = self.nn(x)
        x = self.last_act_fn(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        self.train_batch = batch  # Save for use at the end of the epoch
        x, y, rots = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", float(loss), on_step=True, on_epoch=True)
        return loss

    # def render(self, x, y, y_hat, n=3):
    #     if self.input == "grayscale":
    #         z = torch.vstack((x[:n], y[:n], y_hat[:n]))
    #     elif self.input == "rgb":
    #         z = torch.vstack(
    #             (x[:n], y[:n].repeat(1, 3, 1, 1), y_hat[:n].repeat(1, 3, 1, 1))
    #         )
    #     return torchvision.utils.make_grid(z, nrow=n)

    # def on_train_epoch_end(self):
    #     x, y, rots = self.train_batch
    #     y_hat = self(x)
    #     img = self.render(x, y, y_hat, 2)
    #     self.trainer.logger.experiment.log(
    #         {"train_img": wandb.Image(img), "epoch": self.current_epoch}
    #     )

    def validation_step(self, batch, batch_idx):
        self.val_batch = batch
        x, y, _ = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", float(val_loss))

    # def on_validation_epoch_end(self):
    #     x, y, rots = self.val_batch
    #     y_hat = self(x)
    #     img = self.render(x, y, y_hat)
    #     self.trainer.logger.experiment.log(
    #         {"val_img": wandb.Image(img), "epoch": self.current_epoch}
    #     )
    #     acc = self.accuracy(y_hat, rots)
    #     self.log("accuracy", np.mean(acc))
    #     self.log("not_enough_points", np.sum(np.where(acc == np.pi)))
    #     self.log("median_acc", float(np.median(acc)))
    #     hist = wandb.Histogram(acc)
    #     wandb.log({"acc_dist": hist})

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        # img = self.render(x, y, y_hat)
        # self.trainer.logger.experiment.log(
        #     {"test_img": wandb.Image(img), "epoch": self.current_epoch}
        # )
        test_loss = self.loss_fn(y_hat, y)
        self.log("test_loss", test_loss)

    # def predict_step(self, batch, batch_idx):
    #     x, y, rots = batch
    #     pred = self(x)
    #     return pred

    # def accuracy(self, y_hat, rots):
    #     delta = []
    #     for i in range(rots.shape[0]):
    #         rot_hat = self.estimate_rot(y_hat[i])
    #         if rot_hat is None:
    #             delta.append(np.pi)
    #             continue
    #         rot = R.from_quat(rots[i].cpu().numpy())
    #         delta.append(np.abs((rot_hat * rot.inv()).magnitude()))
    #     delta = np.array(delta)
    #     return delta

    # def dice_metric(self, inputs, target):
    #     intersection = 2.0 * (target * inputs).sum()
    #     union = target.sum() + inputs.sum()
    #     if target.sum() == 0 and inputs.sum() == 0:
    #         return 1.0
    #     return intersection / union

    # def estimate_rot(self, heatmap):
    #     heatmap = T.ToPILImage()(heatmap)
    #     heatmap = np.array(heatmap)
    #     rot, mask, heatmap = sp.heatmap2rot(heatmap, self.geohasher)
    #     return rot

    # def get_dots(self, y_hat):
    #     coord = []
    #     for el in y_hat:
    #         img = el.permute(1, 2, 0).clone().detach().squeeze().cpu().numpy()
    #         peaks = peak_local_max(img)
    #         coord.append(np.argwhere(peaks == True))
    #     return coord

    # def get_gt_dots(self, y):
    #     coords = []
    #     for el in y:
    #         coord = torch.argwhere(el == 1).clone().detach().cpu().numpy()
    #         coord = np.delete(coord, 0, axis=1)
    #         coords.append(coord)
    #     return coords
