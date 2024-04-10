import torch
import torchvision
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
        self.log("train_loss", loss, on_step=True, on_epoch=True)

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
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    

