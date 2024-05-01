import torch
import torchvision
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from modelling.metric import mean_iou  # For IoU metric
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss



class MaskRCNNLightning(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
        self.map_metric = MeanAveragePrecision() 


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss_dict = {k+'_train': v for k, v in loss_dict.items()}
        loss = sum(loss for loss in loss_dict.values())

        # Log for both progress bar and logging
        self.log_dict(loss_dict, on_step=True, on_epoch=True) 
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss_dict = {k+'_val': v for k, v in loss_dict.items()}
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True) 

        # Inference
        self.model.eval()
        outputs = self.model(images)

        # Calculate mAP
        self.map_metric.update(outputs, targets)
        map_value = self.map_metric.compute()
        for k,v in map_value.items():
            if k=="classes":
                continue
            self.log(k,v,on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(
    self.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)
    

class DeepLabV3Lightning(LightningModule):
    def __init__(self, hparams,num_classes=9):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Load DeepLabV3 with ResNet50 backbone
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier[-1] = torch.nn.Conv2d(
                256, num_classes, kernel_size=(1, 1), stride=(1, 1)
            )

        # Mean IoU metric (suitable for segmentation)
        self.iou_metric = mean_iou
        self.loss = BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)["out"]  # DeepLabv3 output format

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)["out"].squeeze()
        loss = self.loss(outputs, targets)

        # Logging
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        print(images.shape)
        print(targets.shape)
        outputs = self.model(images)["out"].squeeze()
        print(outputs.shape)
        loss = self.loss(outputs, targets)

        map_value = sel

        self.

        # Logging
        self.log_dict({'val_loss': loss}, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True) 

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.scheduler_factor, patience=self.scheduler_patience)
        return {"optimizer": optimizer, "monitor": "val_loss", "lr_scheduler": scheduler,}