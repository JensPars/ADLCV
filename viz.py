import os
from torchvision.datasets import CocoDetection
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

import albumentations.pytorch as AT
import albumentations as A

from simple_copy_paste.coco import CocoDetectionCP
from simple_copy_paste.simple_copy_paste import CopyPaste
from argparse import ArgumentParser
import torch
from torch.utils.data import Subset
from instance_copy_paste.dataset import SynData, COCO_DETECTION
from instance_copy_paste.copy_paste import InstanceCopyPaste, InstanceRetriever
from train_mask_rcnn_lightning import plot_images_with_boxes_and_masks, DM

dm = DM(batch_size=2, syn_data=True)
dm.setup()
path2imgs = "/zhome/ca/9/146686/ADLCV/data/data-llm"

for batch in dm.train_dataloader():
    images, targets = batch
    fig = plot_images_with_boxes_and_masks(images, targets, len(images))
    fig.savefig('zzz.png')
    break