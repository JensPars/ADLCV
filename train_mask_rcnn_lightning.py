import yaml
import numpy as np
from glob import glob

from torch.utils.data import Subset
from pytorch_lightning import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from dotenv import load_dotenv
from data import DM
from model import MaskRCNNModel

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from data import DM

load_dotenv()

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        config = self.parser.dump(
            self.config, skip_none=False
        )  # Required for proper reproducibility
        # yaml to dict
        config = yaml.safe_load(config)
        trainer.logger.experiment.config.update({"config": config})


def cli_main():
    cli = LightningCLI(
        MaskRCNNModel,
        DM,
        save_config_kwargs={"overwrite": True},
        save_config_callback=LoggerSaveConfigCallback,
    )


if __name__ == "__main__":
    cli_main()
