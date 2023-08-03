import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from loss import FocalLoss
from torchvision import transforms as T


class Layer(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn=nn.ReLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_dim, affine=True)
        self.act_fn = act_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.act_fn(x)
        x = self.batch_norm(x)
        return x


class DotDetector(pl.LightningModule):
    def __init__(
        self,
        input="rgb",
        layers="64, 64, 128, 256, 256, 128, 64, 64",
        act_fn="leaky_relu",
        last_act_fn="sigmoid",
        loss_fn="focal_loss",
    ):
        super().__init__()

        ## Specifying the model hyperparmeters
        # Type of the input image
        if input == "rgb":
            self.input = "rgb"
            prev_dim = 3
        elif input == "grayscale":
            self.input = "grayscale"
            prev_dim = 1
        else:
            raise ValueError("Wrong input defined")

        # Activation function used
        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "leaky_relu":
            self.act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Activation function not implemented")

        # Activation function used for the last layer
        if last_act_fn == "sigmoid":
            self.last_act_fn = torch.sigmoid
        elif last_act_fn == "hardsigmoid":
            self.last_act_fn = F.hardsigmoid
        else:
            raise ValueError("last_act_fn: {} is not supported".format(last_act_fn))

        # Loss used to train the model
        if loss_fn == "focal_loss":
            self.loss_fn = FocalLoss()
        elif loss_fn == "L1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError("loss_fn: {} not supported".format(loss_fn))

        ## Creating the NN architecture
        self.layers = []
        layer_tuple = tuple(map(int, layers.split(",")))
        for i in range(len(layer_tuple)):
            dim = layer_tuple[i]
            if prev_dim not in (1, 3):
                if prev_dim < dim:
                    self.layers.append(nn.MaxPool2d(2, stride=2, padding=0))
                elif prev_dim > dim:
                    self.layers.append(nn.Upsample(scale_factor=2))
            self.layers.append(Layer(prev_dim, dim, self.act_fn))
            prev_dim = dim
        # Add the last layer that outputs the head
        self.layers.append(nn.Conv2d(prev_dim, 1, 1, padding=0))
        self.nn = nn.Sequential(*self.layers)

        print("Saving hyperparmeters")
        self.save_hyperparameters()

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

    def render(self, x, y, y_hat, n=3):
        if self.input == "grayscale":
            z = torch.vstack((x[:n], y[:n], y_hat[:n]))
        elif self.input == "rgb":
            z = torch.vstack(
                (x[:n], y[:n].repeat(1, 3, 1, 1), y_hat[:n].repeat(1, 3, 1, 1))
            )
        return torchvision.utils.make_grid(z, nrow=n)

    def on_train_epoch_end(self):
        x, y, rots = self.train_batch
        y_hat = self(x)
        img = self.render(x, y, y_hat, 4)
        self.logger.experiment.log(
            {"train_img": wandb.Image(img), "epoch": self.current_epoch}
        )

    def validation_step(self, batch, batch_idx):
        self.val_batch = batch
        x, y, _ = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", float(val_loss))

    def on_validation_epoch_end(self):
        x, y, rots = self.val_batch
        y_hat = self(x)
        img = self.render(x, y, y_hat, 4)
        self.trainer.logger.experiment.log(
            {"val_img": wandb.Image(img), "epoch": self.current_epoch}
        )

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        img = self.render(x, y, y_hat)
        self.trainer.logger.experiment.log(
            {"test_img": wandb.Image(img), "epoch": self.current_epoch}
        )
        test_loss = self.loss_fn(y_hat, y)
        self.log("test_loss", test_loss)

    # def predict_step(self, batch, batch_idx):
    #     x, y, rots = batch
    #     pred = self(x)
    #     return pred
