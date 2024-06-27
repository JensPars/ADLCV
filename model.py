import torch
import lightning.pytorch as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def linear_warmup_cosine_annealing(optimizer, warmup_steps, total_steps, min_lr=3e-5):
    # Linear warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    # Cosine annealing
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps-warmup_steps, eta_min=min_lr)
    # Sequential scheduler
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    return scheduler


class MaskRCNNModel(pl.LightningModule):
    def __init__(self, n_warmup, lr: float = 3e-4, model_type="resnet-maskrcnn", pretrained_backbone=False):
        """
        Initialize the MaskRCNNModel.

        Args:
            lr (float, optional): Learning rate for the optimizer. Defaults to 3e-5.
            model_type (str, optional): Type of the model. Defaults to "resnet-maskrcnn".
            pretrained_backbone (bool, optional): Whether to use a pretrained backbone. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters()
        self.n_warmup = n_warmup
        self.lr = lr
        if model_type == "resnet-maskrcnn":
            if pretrained_backbone:
                self.model = maskrcnn_resnet50_fpn_v2(
                    weights_backbone=ResNet50_Weights.DEFAULT, num_classes=91
                )
            else:
                self.model = maskrcnn_resnet50_fpn_v2(num_classes=91)
                
        elif model_type == "mobilenet-maskrcnn":
            if pretrained_backbone:
                self.model = fasterrcnn_mobilenet_v3_large_fpn(weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
                                                                                            num_classes=91)
            else:
                self.model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=91)
        else:
            raise ValueError(f"Model type {model_type} not recognized.")   
            
        self.map_metric = MeanAveragePrecision()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        losses = self.compute_losses(images, targets)
        self.log("train_loss", losses, on_epoch=True, on_step=False)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)  # Get model predictions
        # update the Mean Average Precision metric
        metric = self.map_metric(outputs, targets)
        self.log("val_mAP", metric['map'], on_epoch=True, on_step=False)


    def compute_losses(self, images, targets):
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)  # Get model predictions
        # update the Mean Average Precision metric
        self.map_metric.update(outputs, targets)
        return outputs

    def on_test_epoch_end(self):
        # compute the Mean Average Precision
        map_value = self.map_metric.compute()
        self.log("map", map_value)
        self.map_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        #scheduler = LinearWarmupCosineAnnealingLR(
        #    optimizer, warmup_epochs=self.n_warmup, max_epochs=self.max_epochs
        #)
        scheduler = linear_warmup_cosine_annealing(optimizer, self.n_warmup, self.trainer.max_steps)
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}

            
        }
