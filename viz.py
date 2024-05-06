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

#load_dotenv()

def data_subset(dataset,fraction):
    # Calculate half of the length of the dataset
    num_samples = int(len(dataset)*fraction)
    # Create a range of indices from 0 to num_samples-1
    indices = range(num_samples)
    return Subset(dataset, indices)

syn_dataset = SynData("data/sdxl-turbo-corrputed", {"car": 2, "bus": 1, "boat": 3})
instance_retriever = InstanceRetriever(syn_dataset)


# Argument parsing
parser = ArgumentParser(description="Choose the COCO dataset version for training.")
parser.add_argument("--copy_paste", help="Use COCOCP dataset for training.")
parser.add_argument("--data_fraction", help="portion of data to use for training", default=1., type=float)
parser.add_argument("--syn_data", help="Use synthetic data for training")
args = parser.parse_args()


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Define transformations
val_transform = transforms.Compose([
    transforms.ToTensor(),
    #A.Resize(512, 512),
])

inst_transform = transform = A.Compose(
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
        format="coco", min_visibility=0.05, label_fields=["labels"]
    ),
)

transform = A.Compose([
    A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
    A.PadIfNeeded(512, 512, border_mode=0), #constant 0 border
    A.Resize(512, 512),
    CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
    AT.ToTensorV2(),

],bbox_params=A.BboxParams(format="coco"))
if args.copy_paste == "True":
    train_dataset = CocoDetectionCP(root="/work3/s194649/train2017", annFile='car_boat_bus_train.json', transforms=transform)
else:
    train_dataset = CocoDetection(root="/work3/s194649/train2017", annFile='car_boat_bus_train.json', transform=val_transform)
    train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset, target_keys=["boxes", "labels", "masks"])

if args.syn_data == "True":
    transform = A.Compose(
    [
        A.RandomScale(
            scale_limit=(-0.9, 1), p=1
        ),  # LargeScaleJitter from scale of 0.1 to 2
        A.PadIfNeeded(
            512, 512, border_mode=0
        ),  # pads with image in the center, not the top left like the paper
        A.Resize(512, 512, p=1),
    ],
    bbox_params=A.BboxParams(
        format="coco", min_visibility=0.05, label_fields=["labels"]
        ),
    )
    train_dataset  = COCO_DETECTION(
    "/work3/s194649/train2017",
    "car_boat_bus_train.json",
    categories=["boat", "car", "bus"],
    transform=transform,
    instance_copy_paste=InstanceCopyPaste(instance_retriever, "random", 8, 0.1, 0),
    )


# sample 100 images and labels from the dataset and display in a 10x10 grid, use torchvision.utils.make_grid
# to create the grid
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

train_dataset = data_subset(train_dataset, args.data_fraction)
# val_dataset = CocoDetection(root="/work3/s194649/val2017", annFile='car_boat_bus_val.json', transform=val_transform)
# val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset, target_keys=["boxes", "labels", "masks"])
plt.imshow(train_dataset[0][0].permute(1, 2, 0))
plt.savefig("zzzzzzz.png")
