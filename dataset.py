import os
import torch
import torchvision
from PIL import Image
import random
import numpy as np

from torch.utils.data import Dataset
from glob import glob
import torchvision

from torchvision.transforms import v2 as T

from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision import transforms, datasets
from torchvision.datasets import CocoDetection
import lightning.pytorch as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
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



class adaptive_pad:
    def __init__(self, h=1024, w=1024):
        self.h = h
        self.w = w
    
    def __call__(self, input):
        img = input[0]
        h, w = img.shape[-2:]
        pad_w = self.h - h
        pad_h = self.w - w
        pad_w = max(pad_w, 0)
        pad_h = max(pad_h, 0)
        padh1 = random.randint(0, pad_h)
        padh2 = pad_h - padh1
        padw1 = random.randint(0, pad_w)
        padw2 = pad_w - padw1
        #pad = (pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2)
        pad = (padh1, padw1, padh2, padw2)
        # pad = (pad_w, pad_h, 0, 0)
        p_t = T.Pad(pad, fill=0, padding_mode="constant")
        return p_t(input)

class PasteDataset(Dataset):
    def __init__(self, dataset1, dataset2, img_sz=1024):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.lsj = T.Compose([T.ScaleJitter((img_sz, img_sz), (0.1, 2.0)), adaptive_pad(img_sz, img_sz),  T.RandomCrop([img_sz,img_sz]), T.SanitizeBoundingBoxes()])
        self.sanitize = T.SanitizeBoundingBoxes()

    def __len__(self):
        return len(self.dataset2)

    def __getitem__(self, idx):
        source = self.dataset1[random.randint(0, len(self.dataset2)-1)]
        source = self.lsj(source)
        target = self.dataset2[idx]
        target = self.lsj(target)
        return self.sanitize(paste(source, target))

def paste(source, target):
    src_img = source[0]
    src_masks = source[1]["masks"].any(axis=0)
    trg_img = target[0]
    trg_masks = target[1]["masks"]
    # Paste the source image on the target image
    trg_img = src_img*src_masks + trg_img*(1-src_masks)
    # calculate mask occlusion
    intersection = trg_masks*src_masks
    trg_masks = trg_masks - intersection
    n_pixels = trg_masks.sum(axis=(1,2))
    propotion_occluded = intersection.sum()/n_pixels
    non_occluded = propotion_occluded < 0.99
    #non_occluded = (source[1]["masks"].sum(axis=(1,2))==0)&non_occluded
    boxes = target[1]["boxes"][non_occluded]
    masks = trg_masks[non_occluded]
    labels = target[1]["labels"][non_occluded]
    # update the target masks, bboxs and labels
    boxes = torch.concatenate([boxes, source[1]["boxes"]], axis=0)
    masks = torch.concatenate([masks, source[1]["masks"]], axis=0)
    labels = torch.concatenate([labels, source[1]["labels"]], axis=0)
    masks = torchvision.tv_tensors.Mask(masks)
    # calculate new boxes from masks
    #breakpoint()
    try:
        boxes = torchvision.ops.masks_to_boxes(masks)
    except:
        print(masks.sum(axis=(1,2)))
    boxes = torchvision.tv_tensors.BoundingBoxes(boxes, format="xyxy", canvas_size=(trg_img.shape[-1], trg_img.shape[-2]))
    return trg_img, {"boxes": boxes, "masks": masks, "labels": labels}

class SubsampleCOCO(CocoDetection):
    def __init__(self, root_dir, annFile_src, transform):
       self.dataset = CocoDetection(root_dir, annFile_src, transform=transform)
       self.dataset = datasets.wrap_dataset_for_transforms_v2(self.dataset, target_keys=["boxes", "labels", "masks"])
        
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        n_instances = random.randint(0, len(target['labels']))
        selected_instances = random.sample(range(len(target['labels'])), n_instances)
        # select random target
        masks = target["masks"][selected_instances]
        labels = target["labels"][selected_instances]
        boxes = target["boxes"][selected_instances]
        masks = torchvision.tv_tensors.Mask(masks)
        boxes = torchvision.tv_tensors.BoundingBoxes(boxes, format="xyxy", canvas_size=(img.shape[-2], img.shape[-1]))
        return img, {"boxes": boxes, "masks": masks, "labels": labels}
    
    def __len__(self):
        return len(self.dataset)

#self.val = CocoDetection(os.environ.get("COCO_DATA_DIR_VAL"), os.environ.get("COCO_VAL_ANN"), transform=transform)
#self.val = datasets.wrap_dataset_for_transforms_v2(self.val, target_keys=["boxes", "labels", "masks"])


class CoCoDet(Dataset):
    def __init__(self, root_dir, annFile, transform=None):
        self.dataset = CocoDetection(root_dir, annFile, transform=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]))
        self.dataset = datasets.wrap_dataset_for_transforms_v2(self.dataset, target_keys=["boxes", "labels", "masks"])
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if len(target) == 0 or "boxes" not in target.keys():
            print(f"{idx} is missing target")
            return self.__getitem__(random.randint(0, len(self.dataset)-1))
        
        if self.transform is not None:
            img, target = self.transform((img, target))
        #print(target.keys())
        #print(target["labels"].unique())
        return img, target

    def __len__(self):
        return len(self.dataset)