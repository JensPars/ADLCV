import torch
import os
from PIL import Image
import random
import numpy as np

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from typing import List, Optional
from glob import glob
import pickle
import zlib
import albumentations as A
import albumentations.pytorch as AT

def read_and_decompress(file):
    # Read the compressed mask from the pickle file
    with open(file, "rb") as f:
        compressed_mask = pickle.load(f)
    # Decompress the mask
    decompressed_mask = zlib.decompress(compressed_mask)
    # Convert the decompressed mask to numpy array
    mask = np.frombuffer(decompressed_mask, dtype=np.bool_)
    n = np.sqrt(len(mask)).astype(int)
    mask = mask.reshape((n, n))
    return mask


class SynData(Dataset):
    """Used in the FID evaluation of generated images."""
 
    def __init__(self, root, cat_dict, fid=False):
        super().__init__()
        cat_dirs = sorted(glob(root + "/*"))
        self.syn_imgs = []
        self.syn_lbls = []
        self.cls = []
        
        for cat_dir in cat_dirs:
            cat = cat_dir.split("/")[-1]
            self.syn_imgs += sorted(glob(cat_dir + "/*.jpg"))
            self.syn_lbls += sorted(glob(cat_dir + "/*.pkl"))
            self.cls += [cat_dict[cat]] * len(sorted(glob(cat_dir + "/*.jpg")))
            
        assert len(self.syn_imgs) == len(self.syn_lbls)
        self.fid = fid


    def get_transform(self):
        img_size = random.randint(64,256)
        return A.Compose([
            A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
            A.PadIfNeeded(img_size, img_size, border_mode=0), #pads with image in the center, not the top left like the paper
            A.Resize(img_size, img_size, p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
        ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05))

    def _get_bbox(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        assert len(np.where(rows)[0]) > 0
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        assert ymax >= ymin and xmax >= xmin
        return [int(xmin), int(ymin), int(xmax)-int(xmin), int(ymax)-int(ymin)]

    def __len__(self):
        return len(self.syn_imgs)
    
    def __getitem__(self, idx):
        output = self._get_syn_data(idx)
        transform = self.get_transform()

        output = transform(**output)
        return output

    def _get_syn_data(self, idx):
        """
        Get synthetic data
        """
        # Load synthetic image
        syn_img = Image.open(self.syn_imgs[idx]).convert("RGB")
        syn_img_class = self.cls[idx]
        # TODO: Map the class to the label
        syn_img = np.array(syn_img)
        # Load synthetic mask
        syn_mask = read_and_decompress(self.syn_lbls[idx])
        syn_mask = syn_mask.astype(np.uint8)
        #syn_bboxes = self._extract_bbox(syn_mask)
        syn_bbox = self._get_bbox(syn_mask)
        syn_bbox = syn_bbox + [self.cls[idx]] # Add label to the bounding box # TODO: Change this to a more general solution
        output = {
            "image": syn_img,
            "masks": [syn_mask],
            "bboxes": [syn_bbox],
            'labels': syn_img_class
        }
        return output

class COCO_DETECTION(Dataset):

    def __init__(self, root: str, anno_file: str, categories: Optional[List[str]] = None, transform=None, min_keypoints_per_image: int = 10, instance_copy_paste = None):
        """
        Args:
            root (str): Root directory of the COCO dataset.
            anno_file (str): Path to the COCO annotation file.
            categories (list, optional): List of categories to include. If not specified, all categories are included.
            transform (callable, optional): A function/transform to apply to the synthetic image and mask.
        """
        super(COCO_DETECTION, self).__init__()
        self.coco = COCO(anno_file)
        self.root = root
        self.transform = transform
        self.instance_copy_paste = instance_copy_paste
        self.min_keypoints_per_image = min_keypoints_per_image
        self.cat_ids = None
        # Filter annotations by category IDs if specified
        # if categories:
        #     self.cat_ids = self.coco.getCatIds(catNms=categories)
        #     img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        # else:
        img_ids = self.coco.getImgIds()

        self.ids = list(sorted(img_ids))
    
    def __len__(self):
        return len(self.ids)
    
    def _get_coco_image(self, idx:int):
        path = os.path.join(self.root, self.coco.loadImgs(idx)[0]['file_name'])
        return Image.open(path).convert('RGB')

    def _get_coco_data(self, idx:int):
        """
        Get COCO data for a random image.

        Args:
            idx (int): Index of the image to retrieve data for.

        Returns:
            dict: A dictionary containing the image, masks, and bounding boxes.
        """
        # Sample instance from COCO dataset
        img_id = self.ids[idx] # Select Random Image from COCO
        # Load image
        img = np.array(self._get_coco_image(img_id))
        # Specific instances in the dataset
        # if self.cat_ids:
        #     ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
        # else:
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # Load annotations
        coco_annotation = self.coco.loadAnns(ann_ids)
        masks = []
        bboxes = []
        labels = []

        for ann in coco_annotation:
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            bboxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        output = {
            "image": img,
            "masks": masks,
            "bboxes": bboxes,
            "labels": labels
        }
        return output

    def post_process(self, img, target):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img.transpose(2, 0, 1))
            
        target = {
            "masks": torch.concat([torch.as_tensor(mask).unsqueeze(0).float() for mask in target["masks"]]),
            "boxes": torch.as_tensor(target["boxes"]).float(),
            "labels": torch.as_tensor(target['labels'], dtype = torch.int64)
        }

        return img.float(), target

    def __getitem__(self, idx):
        coco_data = self._get_coco_data(idx)

        if self.transform:
            coco_data = self.transform(**coco_data)
        # Remove the segmentations masks that are all zeros
        masks = coco_data["masks"]
        masks = [mask for mask in masks if mask.sum() > 0]
        coco_data["masks"] = masks
        # Apply instance copy-paste augmentation
        if self.instance_copy_paste:
            img, target = self.instance_copy_paste(**coco_data)
        else:
            img = coco_data["image"]
            target = {
                "masks": coco_data["masks"],
                "boxes": coco_data["bboxes"],
                "labels": coco_data["labels"]
            }
        img, target = self.post_process(img, target)
        return img, target

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno, min_keypoints_per_image):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False