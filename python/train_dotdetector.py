from dotdetector import DotDetector
from pathlib import Path
import pytorch_lightning as pl
from argparse import ArgumentParser
from dotdatamodule import DotDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.loggers import TensorBoardLogger

if __name__ == "__main__":
    pl.seed_everything(0, workers=True)
    # wandb_logger = WandbLogger(project="dot_detection")
    logger = TensorBoardLogger()

    parser = ArgumentParser(description="Training and model parameters")
    # Add model specific args
    parser = DotDetector.add_model_specific_args(parser)
    # Add all training options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    # Set default values
    parser.set_defaults(
        gpus=1,
        input="rgb",
        logger=logger,
        auto_lr_find=True,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        auto_scale_batch_size="binsearch",
    )
    args = parser.parse_args()
    dict_args = vars(args)
    # parser.print_help()
    # print(dict_args)

    trainer = pl.Trainer.from_argparse_args(args)
    data_dir = Path(
        "/home/gossard/Code/tt_ws/src/tt_tracking_system/tt_spindetection/spin_motor_dots_andro_ball/"
    )
    data = DotDataModule(data_dir, data_aug=True)
    detector = DotDetector(**dict_args)
    trainer.fit(detector, data)
    trainer.test(detector, data)
