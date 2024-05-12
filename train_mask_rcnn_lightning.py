import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision import transforms, datasets
from torchvision.datasets import CocoDetection
from argparse import ArgumentParser
from simple_copy_paste.coco import CocoDetectionCP
from instance_copy_paste.dataset import SynData, COCO_DETECTION
from instance_copy_paste.copy_paste import InstanceCopyPaste, InstanceRetriever
import albumentations as A
import albumentations.pytorch as AT
import lightning.pytorch as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from simple_copy_paste.simple_copy_paste import CopyPaste
from torchvision.models import ResNet50_Weights
from dotenv import load_dotenv
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pytorch_lightning.loggers import Logger
# import 
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import LearningRateMonitor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import make_grid
import wandb
import numpy as np


def plot_images_with_boxes_and_masks(images, targets, num_images=4):
    fig, axs = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
    if num_images == 1:
        axs = [axs]  # Ensure axs is iterable when num_images=1

    for idx in range(min(num_images, len(images))):
        img = images[idx].permute(1, 2, 0).clone().detach().cpu().numpy()  # Convert image from (C, H, W) to (H, W, C)
        axs[idx].imshow(img, cmap='gray')  # Display the background image

        if 'masks' in targets[idx]:
            all_masks = targets[idx]['masks']
            for mask in all_masks:
                mask = mask.squeeze()  # Assuming masks are (1, H, W) and removing singular dimensions
                rgba_mask = np.zeros((*mask.shape, 4))  # Create an RGBA mask

                rgba_mask[:, :, 2] = 1.0  # Blue channel
                mask_data = mask.clone().detach().cpu().numpy()
                rgba_mask[:, :, 3] = mask_data * 0.1  # Alpha channel, scale mask by 0.5 for transparency

                # Overlay the color mask with transparency where mask values are zero
                axs[idx].imshow(rgba_mask)

        # Draw bounding boxes
        for box in targets[idx]['boxes']:
            box = box.cpu().numpy()
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='r', facecolor='none')
            axs[idx].add_patch(rect)

        axs[idx].axis('off')
    plt.tight_layout()
    return fig

load_dotenv()
def data_subset(dataset,fraction):
    # Calculate half of the length of the dataset
    num_samples = int(len(dataset)*fraction)
    # Create a range of indices from 0 to num_samples-1
    indices = range(num_samples)
    return Subset(dataset, indices)

class MaskRCNNModel(pl.LightningModule):
    def __init__(self, lr: float = 3e-5):
        """
        Initialize the MaskRCNNModel.

        Args:
            syn_data (bool, optional): Whether to use synthetic data. Defaults to False.
            copy_paste (bool, optional): Whether to use simple copy paste. Defaults to False.
            data_fraction (float, optional): Fraction of the dataset to use. Defaults to 1.0.
            lr (float, optional): Learning rate for the optimizer. Defaults to 3e-5.
            batch_size (int, optional): Batch size for training and validation. Defaults to 8.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        """
        super().__init__()
        self.lr = lr
        self.model = maskrcnn_resnet50_fpn_v2(weights_backbone=ResNet50_Weights.DEFAULT,num_classes=4)
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.map_metric = MeanAveragePrecision()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        # print(np.unique(images[0].clone().detach().cpu().numpy()))
        losses = self.compute_losses(images, targets)
        
        # Plot and log images each epoch
       # if batch_idx == 0:  # Just log/images for the first batch
       #     fig = plot_images_with_boxes_and_masks(images, targets)
       #     images = wandb.Image(fig, caption="Train Images")
       #     self.log({"train_examples": images})
       #     plt.close(fig)
       # log loss
        self.log('train_loss', losses, on_epoch=True, on_step=False)
        return losses

    def validation_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        losses = self.compute_losses(images, targets)
        self.log('val_loss', losses, on_epoch=True, on_step=False)
        
        # Plot and log images each epoch
#        if batch_idx == 0:  # Just log images for the first batch
#            fig = plot_images_with_boxes_and_masks(images, targets)
#            images = wandb.Image(fig, caption="Validation Images")
#            self.log({"val_examples": images})
#            plt.close(fig)
        
        #return losses

    def compute_losses(self, images, targets):
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        #self.log('train_loss' if self.training else 'val_loss', losses, on_epoch=True, on_step=False)
        return losses
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)   # Get model predictions
        # update the Mean Average Precision metric
        self.map_metric.update(outputs, targets)
        return outputs
    
    def on_test_epoch_end(self):
        # compute the Mean Average Precision
        map_value = self.map_metric.compute()
        self.log('map', map_value)
        self.map_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=self.trainer.max_epochs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    
class DM(pl.LightningDataModule):
    """
    Data module for training Mask R-CNN using PyTorch Lightning.

    Args:
        data_dir (str): Path to the directory containing the data.
        batch_size (int): Batch size for training and validation dataloaders.
        data_fraction (float): Fraction of the dataset to use.
        syn_data (bool): Flag indicating whether to use synthetic data.
        copy_paste (bool): Flag indicating whether to use copy-paste augmentation.
        num_workers (int): Number of workers for data loading.

    Attributes:
        data_dir (str): Path to the directory containing the data.
        batch_size (int): Batch size for training and validation dataloaders.
        data_fraction (float): Fraction of the dataset to use.
        val_transform (torchvision.transforms.Compose): Transformations applied to validation data.
        syn_data (bool): Flag indicating whether to use synthetic data.
        copy_paste (bool): Flag indicating whether to use copy-paste augmentation.
        num_workers (int): Number of workers for data loading.
        instance_retriever (InstanceRetriever): Instance retriever for synthetic data.

    """

    def __init__(self, data_dir: str = 'data/data-llm', batch_size: int = 32, data_fraction: float = 1.0, syn_data: bool = False, copy_paste: bool = False, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_fraction = data_fraction
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.syn_data = syn_data
        self.copy_paste = copy_paste
        self.num_workers = num_workers

        if self.syn_data:
            syn_dataset = SynData(self.data_dir, {"car": 2, "bus": 1, "boat": 3})
            print(len(syn_dataset))
            self.instance_retriever = InstanceRetriever(syn_dataset)

        if self.copy_paste:
            print("Using simple copy paste")
            self.train_transform = A.Compose([
                A.RandomScale(scale_limit=(-0.9, 1), p=1),
                A.PadIfNeeded(512, 512, border_mode=0),
                A.Resize(512, 512),
                CopyPaste(blend=True, sigma=1.0, pct_objects_paste=0.5, p=1),
                AT.ToTensorV2(),
            ], bbox_params=A.BboxParams(format="coco"))
        
        elif self.syn_data:
            print("Using Synthetic Data")
            self.train_transform = A.Compose(
                    [
                        A.RandomScale(
                            scale_limit=(-0.9, 1), p=1
                        ),
                        A.PadIfNeeded(
                            512, 512, border_mode=0
                        ),
                    ],
                        bbox_params=A.BboxParams(
                            format="coco", min_visibility=0.05, label_fields=["labels"]),
                        )
        else:
            print("No augmentation")
            self.train_transform = self.val_transform
    
    def _get_dataset(self, train=True):
        """
        Get the dataset based on the train flag.

        Args:
            train (bool): Flag indicating whether to get the training dataset.

        Returns:
            torch.utils.data.Dataset: The dataset.

        """
        root_dir = os.environ.get("COCO_DATA_DIR_TRAIN" if train else "COCO_DATA_DIR_VAL")
        ann_file = 'car_boat_bus_train.json' if train else 'car_boat_bus_val.json'
        if self.copy_paste and train:
            dataset = CocoDetectionCP(root=root_dir, annFile=ann_file, transforms=self.train_transform)
            return data_subset(dataset, self.data_fraction)
        elif self.syn_data and train:
            dataset = COCO_DETECTION(
                os.environ.get("COCO_DATA_DIR_TRAIN"),
                "car_boat_bus_train.json",
                categories=["boat", "car", "bus"],
                transform=self.train_transform,
                instance_copy_paste=InstanceCopyPaste(self.instance_retriever, "random", 8, 0.5, 0),
                )
            return data_subset(dataset, self.data_fraction)
        else:
            dataset = CocoDetection(root=root_dir, annFile=ann_file, transform=self.val_transform)
            dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels", "masks"])
            return dataset
        
        
    def setup(self, stage=None):
        """
        Setup the data module.

        Args:
            stage (str, optional): Stage of the training process. Defaults to None.

        """
        if stage == 'fit' or stage is None:
            self.train = self._get_dataset(train=True)
            self.val = self._get_dataset(train=False)
        if stage == 'test' or stage is None:
            self.test = self._get_dataset(train=False)
    
    def train_dataloader(self):
        """
        Get the dataloader for training.

        Returns:
            torch.utils.data.DataLoader: The training dataloader.

        """
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x)))
    
    def val_dataloader(self):
        """
        Get the dataloader for validation.

        Returns:
            torch.utils.data.DataLoader: The validation dataloader.

        """
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x)))
    
    def test_dataloader(self):
        """
        Get the dataloader for testing.

        Returns:
            torch.utils.data.DataLoader: The testing dataloader.

        """
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x)))
    
        
        

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.config.update({"config": config})
            print("Config saved to logger")

def cli_main():
    cli = LightningCLI(MaskRCNNModel,
                       DM,
                       save_config_kwargs={"overwrite": True},
                       save_config_callback=LoggerSaveConfigCallback)
    
if __name__ == '__main__':
    cli_main()