import pytorch_lightning as pl
from modelling.model import MaskRCNNLightning
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
from modelling.dm import CocoDataModule
import sys
import os
import albumentations as A
from simple_copy_paste.simple_copy_paste import CopyPaste
import wandb
from torchvision.transforms import v2

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

if __name__ == "__main__":
    num_classes = 9  # Number of classes in the dataset

        # ModelCheckpoint setup
    checkpoint_callback = ModelCheckpoint(
        dirpath='/work3/s194633/ADLCV-weights',  # Set directory to save models
        filename='best-maskrcnn-no-copypaste',  # Example filename
        monitor='val_loss',  # Monitor validation loss
        mode='min',          # Save when validation loss decreases
        save_top_k=1,        # Save only the single best model
    )


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
        # A.HorizontalFlip(p=0.5),
        CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1)
        ], bbox_params=A.BboxParams(format="coco",min_visibility=0.05)
    )

    # Dataset setup 
    data_module = CocoDataModule(
        train_dir="/work3/s194633/train2017",
        train_file='coco_subset_annotations.json', 
        val_dir='/work3/s194633/val2017', 
        val_file='coco_subset_annotations_val.json', 
        transforms=None, 
        batch_size=hparams['batch_size']
    )

    # Model
    model = MaskRCNNLightning(hparams)


    wandb.init(name = "no-cp", project="mask-rcnn")  # Initialize wandb project

    trainer = pl.Trainer(
        accelerator="gpu", 
        max_epochs=20, 
        logger=pl.loggers.WandbLogger(),  
        callbacks=[checkpoint_callback],  # Add the callback
        precision=16
    ) 
    trainer.fit(model, datamodule=data_module)  # Fit the model