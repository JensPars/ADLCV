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
#from cmmd import mmd, ClipEmbeddingModel
from pycocotools import coco
from generation_pipeline.cmmd import ClipEmbeddingModel


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

        return masked_images


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
    #[100, 200, 300, 400, 500, 600, 700, 800, 900, 998]:
    root='/work3/s194649/val2017'
    annFile='/work3/s194649/annotations/instances_val2017.json'
    transform = T.Compose([T.Resize([256, 256]),T.ToTensor()])
    coco = coco.COCO(annFile)
    cats = [coco.cats[id]['name'] for id in coco.cats]
    from transformers import AutoProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", do_rescale=False)
    for cat in cats: 
        save_dir = "img_embeds/" + cat
        os.makedirs(save_dir, exist_ok=True)
        realdata = DataLoader(MaskedData(root, annFile, transform=transform, categories=cat),
                              batch_size=16,
                              drop_last=False,
                              collate_fn=lambda x: np.concatenate(x, axis=0))
        n_imgs = 0
        for img in realdata:
            inputs = processor(images=img, return_tensors="pt")
            image_features = model.get_image_features(**inputs)
            n_imgs += img.shape[0]
            # write to npy
            np.save(f"{save_dir}/image_features_{n_imgs}.npy", image_features.detach().cpu().numpy())
            print(n_imgs)
            if n_imgs > 1000:
                break
            
            
            

