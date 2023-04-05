from dotdetector import DotDetector
from dotdatamodule import DotDataModule
from pytorch_lightning.cli import LightningCLI


def cli_main():
    cli = LightningCLI(DotDetector, DotDataModule, save_config_overwrite=True)


if __name__ == "__main__":
    cli_main()
