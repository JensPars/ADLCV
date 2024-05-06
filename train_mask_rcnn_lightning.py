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
from simple_copy_paste.simple_copy_paste import CopyPaste
from torchvision.models import ResNet50_Weights
from dotenv import load_dotenv

load_dotenv()
def data_subset(dataset,fraction):
    # Calculate half of the length of the dataset
    num_samples = int(len(dataset)*fraction)
    # Create a range of indices from 0 to num_samples-1
    indices = range(num_samples)
    return Subset(dataset, indices)

class MaskRCNNModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = maskrcnn_resnet50_fpn_v2(eights_backbone=ResNet50_Weights.DEFAULT,num_classes=4)
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.init_transforms()

    def init_transforms(self):
        if self.args.copy_paste == "True":
            self.train_transform = A.Compose([
                A.RandomScale(scale_limit=(-0.9, 1), p=1),
                A.PadIfNeeded(512, 512, border_mode=0),
                A.Resize(512, 512),
                CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
                AT.ToTensorV2(),
            ], bbox_params=A.BboxParams(format="coco"))
        else:
            self.train_transform = self.val_transform

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses)
        return losses

    def validation_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def train_dataloader(self):
        train_dataset = self._get_dataset(train=True)
        return DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    def val_dataloader(self):
        val_dataset = self._get_dataset(train=False)
        return DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    def _get_dataset(self, train=True):
        # Dataset fetching logic depending on the flag
        root_dir = os.environ.get("COCO_DATA_DIR_TRAIN" if train else "COCO_DATA_DIR_VAL")
        ann_file = 'car_boat_bus_train.json' if train else 'car_boat_bus_val.json'
        if self.args.copy_paste == "True" and train:
            dataset = CocoDetectionCP(root=root_dir, annFile=ann_file, transforms=self.train_transform)
        else:
            dataset = CocoDetection(root=root_dir, annFile=ann_file, transform=self.val_transform)
            dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels", "masks"])
        return data_subset(dataset, self.args.data_fraction)

# Parsing Arguments and Initialize components
parser = ArgumentParser()
parser.add_argument("--copy_paste", type=str, default="False")
parser.add_argument("--data_fraction", type=float, default=1.0)
args = parser.parse_args()

# Trainer setup
logger = WandbLogger(project="copy-paste-project", log_model="all")
checkpoint_callback = ModelCheckpoint(dirpath="./model_checkpoints", save_top_k=1, monitor="val_loss")
trainer = Trainer(max_epochs=50, logger=logger, callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch')],accelerator="cpu")

# Model instantiation and training
model = MaskRCNNModel(args)
trainer.fit(model)