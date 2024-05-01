if __name__ == "__main__":
    from setupHF_cache import *
import os
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms as T
from tqdm import tqdm
from glob import glob
from autosam import read_and_decompress
from cmmd import mmd, ClipEmbeddingModel


class SynData(Dataset):
    """Used in the FID evaluation of generated images."""

    def __init__(self, img_dir, anno_dir, fid=False):
        super().__init__()
        self.imgs = sorted(glob(img_dir + "/*.jpg"))
        self.lbls = sorted(glob(anno_dir + "/*.pkl"))
        assert len(self.imgs) == len(self.lbls)     
        self.fid = fid

    def __getitem__(self, index):
        img = self.imgs[index]
        img = Image.open(img)
        lbl = read_and_decompress(self.lbls[index])
        if self.fid:
            # Set background to black
            img = np.array(img)
            img[~lbl] = 0
            img = Image.fromarray(img)
            img = T.ToTensor()(img) #* 255
            return img#.to(torch.uint8)
        else:
            return img, lbl

    def __len__(self):
        return len(self.imgs)


class MaskedData(Dataset):
    """Used in the FID evaluation of generated images."""

    def __init__(self, root, annotation, transform=None, categories=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.coco = COCO(annotation)
        self.cat_ids = None  # To store category IDs of interest

        if categories:
            self.cat_ids = self.coco.getCatIds(catNms=categories)
            img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        else:
            img_ids = self.coco.getImgIds()

        self.ids = list(sorted(img_ids))

    def _get_image(self, idx):
        path = os.path.join(self.root, self.coco.loadImgs(idx)[0]["file_name"])
        return Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.ids)

    def _crop_image(self, image, bbox):
        x, y, w, h = bbox
        return image.crop((x, y, x + w, y + h))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        if self.cat_ids:
            # Filter annotations by category IDs if specified
            ann_ids = self.coco.getAnnIds(
                imgIds=img_id, catIds=self.cat_ids, iscrowd=None
            )
        else:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

        coco_annotation = self.coco.loadAnns(ann_ids)
        img = self._get_image(img_id)
        objects = []

        for ann in coco_annotation:
            mask = self.coco.annToMask(ann)  # Ensure mask is in 0-255 range
            cropped_img = self._crop_image(img, ann["bbox"])
            cropped_mask = self._crop_image(
                Image.fromarray(mask.astype(np.uint8)), ann["bbox"]
            )

            cropped_masked_img = (
                np.array(cropped_img) * np.array(cropped_mask)[:, :, None]
            )
            cropped_masked_img = Image.fromarray(cropped_masked_img.astype(np.uint8))

            if self.transform:
                cropped_masked_img = self.transform(cropped_masked_img)
            objects.append(cropped_masked_img)

        masked_images = torch.stack(objects, dim=0) if objects else torch.empty(0)

        return masked_images  # Mu


class FID:

    def __init__(self, feature_map: int, syn_data, real_data, device: str, **kwargs):
        """
        args:
        - feature_map: int, number of features to extract from the inception network
            choices [64, 192, 768, 2048]
        """
        assert feature_map in [
            64,
            192,
            768,
            2048,
        ], "feature_map must be one of [64, 192, 768, 2048]"

        self.syn_data = syn_data  # Might be dataset instance or dataloader
        self.real_data = real_data  # Might be dataset instance or dataloader
        self.device = device
        self.feature_map = feature_map

        self.fid = FrechetInceptionDistance()

    def _update(self, dataset, real: bool = False):
        for idx, batch in enumerate(dataset):
            batch = batch.to(self.device)
            self.fid.update(batch, real=real)

    def compute(self):
        self._update(self.syn_data, real=False)
        self._update(self.real_data, real=True)
        return self.fid.compute()

if __name__ == "__main__":
    syndata = SynData(
            img_dir="data/stable-diffusion-xl-base-1.0/car",
            anno_dir="data/stable-diffusion-xl-base-1.0/car",
            fid=True,
        )
    syndata = DataLoader(syndata, batch_size=16, drop_last=False)
    root = "/work3/s194649/val2017"
    anno = "car_boat_bus_val.json"
    transform = T.Compose([T.Resize([512, 512]),T.ToTensor()])
    realdata = DataLoader(MaskedData(root, anno, transform=transform, categories="car"), batch_size=16, drop_last=False)
    fid = FrechetInceptionDistance(feature=2048).to("cuda")
    #for j, (real, syn) in enumerate(zip(realdata, syndata)):
#            syn = (syn*255).to(torch.uint8).to("cuda")
#            real = real.to("cuda")
#            fid.update(real, real=True)
#            fid.update(syn, real=False)
#            if j == n_batchs:
#                break
#            
#        print(f"with n = {i} FID: {fid.compute()}")
    




#if __name__ == "__main__":
#    for i in [500]:
#        syndata = SynData(
#            img_dir="data/experiments_cg/7.5/cat",
#            anno_dir="data/experiments_cg/7.5/cat",
#            fid=True,
#        )
#        n_batchs = i//16 + 1
#        syndata = DataLoader(syndata, batch_size=16, drop_last=False)
#        root = "/work3/s194649/train2017"
#        anno = "cat_subset_annotations_train.json"
#        transform = T.Compose([T.Resize([512, 512]),T.ToTensor()])
#        realdata = DataLoader(MaskedData(root, anno, transform=transform, categories="cat"), batch_size=16, drop_last=False)
#        fid = FrechetInceptionDistance(feature=2048).to("cuda")
#        
#        for j, (real, syn) in enumerate(zip(realdata, syndata)):
#            syn = (syn*255).to(torch.uint8).to("cuda")
#            real = real.to("cuda")
#            fid.update(real, real=True)
#            fid.update(syn, real=False)
#            if j == n_batchs:
#                break
#            
#        print(f"with n = {i} FID: {fid.compute()}")

