import torch
import torchvision
from pytorch_lightning import LightningModule

class MaskRCNNLightning(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        # Log for both progress bar and logging
        self.log_dict(loss_dict, on_step=True, on_epoch=True) 
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict, on_step=True, on_epoch=True) 
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    

