from dotdetector import DotDetector
from pathlib import Path
import pytorch_lightning as pl
from argparse import ArgumentParser
from dotdatamodule import DotDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.cli import LightningCLI


def cli_main():
    cli = LightningCLI(DotDetector, DotDataModule)


if __name__ == "__main__":
    cli_main()
