import pytorch_lightning as pl
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from modelling.model import MaskRCNNLightning
from modelling.dm import CocoDataModule
import sys
import os
import albumentations as A
from simple_copy_paste.simple_copy_paste import CopyPaste
import wandb

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

if __name__ == "__main__":
    num_classes = 9  # Number of classes in the dataset


    # Hyperparameters 
    hparams = {
        'lr': 0.001,
        'batch_size': 8
    }

    # Example with basic Albumentations transforms - adjust as needed
    transform = A.Compose([
        A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
        A.PadIfNeeded(256, 256, border_mode=0), #constant 0 border
        A.RandomCrop(256, 256),
        A.HorizontalFlip(p=0.5),
        CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1)
        ], bbox_params=A.BboxParams(format="coco",min_visibility=0.05)
    )

    # Dataset setup 
    data_module = CocoDataModule(
        root_dir='data/val2017', 
        ann_file='coco_subset_annotations_val.json', 
        transforms=transform, 
        batch_size=hparams['batch_size']
    )

    # Model
    model = MaskRCNNLightning(hparams)


    wandb.init(project="mask-rcnn")  # Initialize wandb project

    trainer = pl.Trainer(accelerator="cpu", max_epochs=10, logger=pl.loggers.WandbLogger())  # Use WandbLogger
    trainer.fit(model, datamodule=data_module)  # Fit the model