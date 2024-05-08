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
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI
from simple_copy_paste.simple_copy_paste import CopyPaste
from torchvision.models import ResNet50_Weights
from dotenv import load_dotenv
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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
                rgba_mask[:, :, 3] = mask_data * 0.5  # Alpha channel, scale mask by 0.5 for transparency

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

class MaskRCNNModel(LightningModule):
    def __init__(self, syn_data=False, copy_paste=False, data_fraction=1.0, lr=3e-5, batch_size=8, num_workers=0):
        super().__init__()
        self.syndata = syn_data
        self.copy_paste = copy_paste
        self.data_fraction = data_fraction
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = maskrcnn_resnet50_fpn_v2(weights_backbone=ResNet50_Weights.DEFAULT,num_classes=4)
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.map_metric = MeanAveragePrecision()
        self.init_transforms()

        # Create instance retriever if using synthetic data
        if self.syn_data:
            syn_dataset = SynData("sdxl-turbo", {"car": 2, "bus": 1, "boat": 3})
            self.instance_retriever = InstanceRetriever(syn_dataset)

    def init_transforms(self):
        if self.copy_paste:
            print("Using simple copy paste")
            self.train_transform = A.Compose([
                A.RandomScale(scale_limit=(-0.9, 1), p=1),
                A.PadIfNeeded(512, 512, border_mode=0),
                A.Resize(512, 512),
                CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
                AT.ToTensorV2(),
            ], bbox_params=A.BboxParams(format="coco"))
        elif self.syn_data:
            print("Using Synthetic Data")
            self.train_transform = A.Compose(
                    [
                        A.RandomScale(
                            scale_limit=(-0.9, 1), p=1
                        ),  # LargeScaleJitter from scale of 0.1 to 2
                        A.PadIfNeeded(
                            512, 512, border_mode=0
                        ),  # pads with image in the center, not the top left like the paper
                        A.RandomCrop(512, 512, p=1),
                    ],
                        bbox_params=A.BboxParams(
                            format="coco", min_visibility=0.05, label_fields=["labels"]),
                        )
        else:
            print("No augmentation")
            self.train_transform = self.val_transform

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        print(np.unique(images[0].clone().detach().cpu().numpy()))
        losses = self.compute_losses(images, targets)
        
        # Plot and log images each epoch
        if batch_idx == 0:  # Just log/images for the first batch
            fig = plot_images_with_boxes_and_masks(images, targets)
            images = wandb.Image(fig, caption="Train Images")
            self.log({"train_examples": images})
            plt.close(fig)
        
        return losses

    def validation_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        losses = self.compute_losses(images, targets)
        
        # Plot and log images each epoch
        if batch_idx == 0:  # Just log images for the first batch
            fig = plot_images_with_boxes_and_masks(images, targets)
            images = wandb.Image(fig, caption="Validation Images")
            self.log({"val_examples": images})
            plt.close(fig)
        
        return losses

    def compute_losses(self, images, targets):
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss' if self.training else 'val_loss', losses, on_epoch=True, on_step=False)
        return losses
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)   # Get model predictions
        # update the Mean Average Precision metric
        self.map_metric.update(outputs, targets)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def train_dataloader(self):
        train_dataset = self._get_dataset(train=True)
        print("Length of train:", len(train_dataset))
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x)))

    def val_dataloader(self):
        val_dataset = self._get_dataset(train=False)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x)))
    
    def test_dataloader(self):
        test_dataset = self._get_dataset(train=False)  # Assuming same settings for val and test for simplicity
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x)))

    def _get_dataset(self, train=True):
        # Dataset fetching logic depending on the flag
        root_dir = os.environ.get("COCO_DATA_DIR_TRAIN" if train else "COCO_DATA_DIR_VAL")
        ann_file = 'car_boat_bus_train.json' if train else 'car_boat_bus_val.json'
        if self.copy_paste and train:
            dataset = CocoDetectionCP(root=root_dir, annFile=ann_file, transforms=self.train_transform)
        elif self.syn_data and train:
            dataset = COCO_DETECTION(
                os.environ.get("COCO_DATA_DIR_TRAIN"),
                "car_boat_bus_train.json",
                categories=["boat", "car", "bus"],
                transform=self.train_transform,
                instance_copy_paste=InstanceCopyPaste(self.instance_retriever, "random", 8, 0.5, 0),
                )
        else:
            dataset = CocoDetection(root=root_dir, annFile=ann_file, transform=self.val_transform)
            dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels", "masks"])
        return data_subset(dataset, self.data_fraction)

def cli_main():
    cli = LightningCLI(MaskRCNNModel)
    
# Parsing Arguments and Initialize components
parser = ArgumentParser()
parser.add_argument("--copy_paste", type=bool, default=False)
parser.add_argument("--data_fraction", type=float, default=1.)
parser.add_argument("--syn_data", type=bool, default=False)
parser.add_argument("--lr", type=float, default=3e-5)
args = parser.parse_args()

# Trainer setup
logger = WandbLogger(project="copy-paste-project", log_model="all")
checkpoint_callback = ModelCheckpoint(dirpath="./model_checkpoints", save_top_k=1, monitor="val_loss")
trainer = Trainer(max_epochs=100, logger=logger, callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch')],accelerator="auto")

# Model instantiation and training
model = MaskRCNNModel(args)
trainer.fit(model)
trainer.test(ckpt_path="best")  # Running testing after training
