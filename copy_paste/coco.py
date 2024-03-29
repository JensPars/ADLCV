import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import numpy as np
import albumentations as A
from copy_paste.helpers import CopyPasteAugmentation
import os
#torchvision transforms
from torchvision.transforms import functional as F

class COCODataset(Dataset):
    def __init__(self, annotation_file, image_dir, transforms=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        for ann in annotations:
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(ann['category_id'])
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        masks = np.array(masks, dtype=np.uint8)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels, masks=masks)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
            masks = np.array(transformed['masks'], dtype=np.uint8)

        return image, boxes, labels, masks
