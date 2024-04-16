import pytorch_lightning as pl
from modelling.model import MaskRCNNLightning, DeepLabV3Lightning
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
from modelling.dm import CocoDataModule
import sys
import os
import albumentations as A
from simple_copy_paste.simple_copy_paste import CopyPaste
import wandb
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

def get_transform(image=True):
    transforms = []
    transforms.append(T.ToTensor())
    if image:
        transforms.append(T.Resize((256, 256), InterpolationMode.BILINEAR))
    else:
        transforms.append(T.Resize((256, 256), InterpolationMode.NEAREST))
    return T.Compose(transforms)

if __name__ == "__main__":
    num_classes = 1  # Number of classes in the dataset

        # ModelCheckpoint setup
    checkpoint_callback = ModelCheckpoint(
        dirpath='models',  # Set directory to save models
        filename='best-seg-no-copypaste',  # Example filename
        monitor='val_loss',  # Monitor validation loss
        mode='min',          # Save when validation loss decreases
        save_top_k=1,        # Save only the single best model
    )


    # Hyperparameters 
    hparams = {
        'lr': 2e-5,
        'batch_size': 2
    }

    # # Example with basic Albumentations transforms - adjust as needed
    # transform = A.Compose([
    #     A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
    #     A.PadIfNeeded(256, 256, border_mode=0), #constant 0 border
    #     A.RandomCrop(256, 256),
    #     # A.HorizontalFlip(p=0.5),
    #     CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1)
    #     ], bbox_params=A.BboxParams(format="coco",min_visibility=0.05)
    # )

    # Dataset setup 
    data_module = CocoDataModule(
        train_dir="data/val2017",
        train_file='bear_subset_annotations_val.json', 
        val_dir='data/val2017', 
        val_file='bear_subset_annotations_val.json', 
        transforms_im=get_transform(image=True),
        transforms_mask=get_transform(image=False), 
        batch_size=hparams['batch_size'],

    )

    # Model
    model = DeepLabV3Lightning(hparams, num_classes=num_classes)


    wandb.init(name = "no-cp-segmentation", project="segmentation-copy-paste")  # Initialize wandb project

    trainer = pl.Trainer(
        accelerator="cpu", 
        max_epochs=20, 
        logger=pl.loggers.WandbLogger(),  
        callbacks=[checkpoint_callback],  # Add the callback
        precision=16,
        profiler="advanced"
    ) 
    trainer.fit(model, datamodule=data_module)  # Fit the model