import random
import string
import torch
import torchvision
import lightning.pytorch as pl
import numpy as np
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision.models.detection.transform import resize_boxes

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
                print("Using ResNet50 backbone /w pretrained weights.")
            else:
                self.model = maskrcnn_resnet50_fpn_v2(num_classes=91)
                print("Using ResNet50 backbone /wo pretrained weights.")
                
        elif model_type == "mobilenet-maskrcnn":
            if pretrained_backbone:
                self.model = fasterrcnn_mobilenet_v3_large_fpn(weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
                                                                                            num_classes=91)
                print("Using MobileNetV3 backbone /w pretrained weights.")
            else:
                self.model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=91)
                print("Using MobileNetV3 backbone /wo pretrained weights.")
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
        raw_images, targets = batch
        original_image_sizes = []
        for img in raw_images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        outputs = self.forward(raw_images)  # Get model predictions
        pred_boxes = [output['boxes'] for output in outputs]
        images,_ = self.model.transform(raw_images, targets)
        features = self.model.backbone(images.tensors)
        boxes, losses = self.model.rpn(images, features)
        #detections = self.model.transform.postprocess(boxes, images.image_sizes, original_image_sizes)
        # calculate the iou between the predicted boxes and the targets
        #iou = self.model.rpn.iou(boxes, targets)
        for img, box, trg, img_shp, org_shp, pred_box in zip(raw_images, boxes, targets, images.image_sizes, original_image_sizes, pred_boxes):
            box = resize_boxes(box, img_shp, org_shp)
            iou = torchvision.ops.box_iou(box, trg['boxes'])
           # print(torchvision.ops.box_iou(box, pred_box).max())
           # print(iou.max())
            iou = iou.max(axis=1)[0].unsqueeze(1)
            # save boxes and iou to txt
            boxiou = torch.cat((box, iou),1)
            # use random name, by generating random string
            random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            # convert to numpy save boxiou
            boxiou = boxiou.cpu().detach().numpy()
            np.savetxt(f'image_outputs/boxiou_{random_string}.txt', boxiou)
            #torch.save(boxiou, f'image_outputs/boxiou_{random_string}.pt')
            # image to PIL
            torchvision.transforms.ToPILImage()(img).save(f'image_outputs/img_{random_string}.png')
            
            

    #def on_test_epoch_end(self):
    #    # compute the Mean Average Precision
    #    map_value = self.map_metric.compute()
    #    self.log("map", map_value)
    #    self.map_metric.reset()
    
        

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
