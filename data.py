import os
import torch
import torchvision
import random

from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision import transforms, datasets
from torchvision.datasets import CocoDetection
import lightning.pytorch as pl
from dotenv import load_dotenv

# import
from torchvision.utils import make_grid
import numpy as np
from utils import vis_image_mask, id_generator

load_dotenv()


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
        # pad = (pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2)
        pad = (padh1, padw1, padh2, padw2)
        # pad = (pad_w, pad_h, 0, 0)
        p_t = T.Pad(pad, fill=0, padding_mode="constant")
        return p_t(input)


class PasteDataset(Dataset):
    def __init__(self, dataset1, dataset2, img_sz=1024):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.lsj = T.Compose(
            [
                T.ScaleJitter((img_sz, img_sz), (0.1, 2.0)),
                adaptive_pad(img_sz, img_sz),
                T.RandomCrop([img_sz, img_sz]),
                T.SanitizeBoundingBoxes(),
            ]
        )
        self.sanitize = T.SanitizeBoundingBoxes()

    def __len__(self):
        return len(self.dataset2)

    def __getitem__(self, idx):
        source = self.dataset1[random.randint(0, len(self.dataset2) - 1)]
        source = self.lsj(source)
        target = self.dataset2[idx]
        target = self.lsj(target)
        return self.sanitize(paste(source, target))


class PasteDataset2(Dataset):
    def __init__(self, dataset1, dataset2, img_sz=1024):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.lsj = T.Compose(
            [
                T.ScaleJitter((img_sz, img_sz), (0.1, 2.0)),
                adaptive_pad(img_sz, img_sz),
                T.RandomCrop([img_sz, img_sz]),
                T.SanitizeBoundingBoxes(),
            ]
        )
        self.sanitize = T.SanitizeBoundingBoxes()

    def __len__(self):
        return len(self.dataset2)

    def __getitem__(self, idx):
        random_instance = random.randint(0, len(self.dataset2) - 1)
        n_instances = len(self.dataset2[random_instance][1]["labels"])
        sources = [
            self.dataset1[random.randint(0, len(self.dataset1) - 1)]
            for i in range(n_instances)
        ]
        target = self.dataset2[idx]
        target = self.lsj(target)
        for i, source in enumerate(sources):
            source = self.lsj(source)
            target = paste(source, target)
        return self.sanitize(target)


def paste(source, target):
    src_img = source[0]
    src_masks = source[1]["masks"].any(axis=0)
    trg_img = target[0]
    trg_masks = target[1]["masks"]
    # Paste the source image on the target image
    trg_img = src_img * src_masks + trg_img * (1 - src_masks)
    # calculate mask occlusion
    intersection = trg_masks * src_masks
    n_pixels_before = trg_masks.sum(axis=(1, 2))
    trg_masks = trg_masks - intersection
    n_pixels = trg_masks.sum(axis=(1, 2))
    propotion_occluded = (n_pixels_before - n_pixels) / n_pixels_before
    #print(propotion_occluded)
    #assert max(propotion_occluded) <= 1
    non_occluded = propotion_occluded < 0.95
    # non_occluded = (source[1]["masks"].sum(axis=(1,2))==0)&non_occluded
    boxes = target[1]["boxes"][non_occluded]
    masks = trg_masks[non_occluded]
    labels = target[1]["labels"][non_occluded]
    # update the target masks, bboxs and labels
    boxes = torch.concatenate([boxes, source[1]["boxes"]], axis=0)
    masks = torch.concatenate([masks, source[1]["masks"]], axis=0)
    labels = torch.concatenate([labels, source[1]["labels"]], axis=0)
    masks = torchvision.tv_tensors.Mask(masks)
    # calculate new boxes from masks
    try:
        boxes = torchvision.ops.masks_to_boxes(masks)
    except:
        print(masks.sum(axis=(1, 2)))
    boxes = torchvision.tv_tensors.BoundingBoxes(
        boxes, format="xyxy", canvas_size=(trg_img.shape[-1], trg_img.shape[-2])
    )
    return trg_img, {"boxes": boxes, "masks": masks, "labels": labels}

def pastebatched(sources, targets):
    '''
    Takes a batch of sources and targets and pastes the sources on the targets
    sources: list of tuples of image and target
    targets: list of tuples of image and target
    '''
    for i in range(len(targets)):
        # sample random source
        source = sources[random.randint(0, len(sources) - 1)]
        # sample max n_instances from target
        n_instances = random.randint(0, len(source[1]['labels']))
        source = sampleinstances(sources, n_instances)
        target = targets[i]
        for src in source:
            target = paste(src, target)
        
        targets[i] = target
    
    return targets
    
    
def sampleinstances(sources, n_instances):
    '''
    Takes a batch of sources and samples n_instances from them
    '''
    selected_imgs = [random.randint(0, len(sources) - 1) for _ in range(n_instances)]
    out = []
    for i in selected_imgs:
        img, trg = sources[i]
        instance = random.randint(0, len(trg["labels"]))
        out.append((img, {"boxes": trg["boxes"][instance].unsqueeze(0),
                          "masks": trg["masks"][instance].unsqueeze(0),
                          "labels": trg["labels"][instance].unsqueeze(0)}))
        
    return out

class SubsampleCOCO(CocoDetection):
    def __init__(self, root_dir, annFile_src, transform, sample_n=None): #propotion_occluded
        self.dataset = CocoDetection(root_dir, annFile_src, transform=transform)
        self.dataset = datasets.wrap_dataset_for_transforms_v2(
            self.dataset, target_keys=["boxes", "labels", "masks"]
        )
        self.sample_n = sample_n

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.sample_n is None:
            n_instances = random.randint(0, len(target["labels"]))
        else:
            n_instances = self.sample_n
        selected_instances = random.sample(range(len(target["labels"])), n_instances)
        # select random target
        masks = target["masks"][selected_instances]
        labels = target["labels"][selected_instances]
        boxes = target["boxes"][selected_instances]
        masks = torchvision.tv_tensors.Mask(masks)
        boxes = torchvision.tv_tensors.BoundingBoxes(
            boxes, format="xyxy", canvas_size=(img.shape[-2], img.shape[-1])
        )
        return img, {"boxes": boxes, "masks": masks, "labels": labels}

    def __len__(self):
        return len(self.dataset)


        
        

class CoCoDet(Dataset):
    def __init__(self, root_dir, annFile, transform=None):
        self.dataset = CocoDetection(
            root_dir,
            annFile,
            transform=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        )
        self.dataset = datasets.wrap_dataset_for_transforms_v2(
            self.dataset, target_keys=["boxes", "labels", "masks"]
        )
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if len(target) == 0 or "boxes" not in target.keys():
            print(f"{idx} is missing target")
            return self.__getitem__(random.randint(0, len(self.dataset) - 1))

        if self.transform is not None:
            img, target = self.transform((img, target))

        return img, target

    def __len__(self):
        return len(self.dataset)


class DM(pl.LightningDataModule):
    """
    Data module for training Mask R-CNN using PyTorch Lightning.

    Args:
        data_dir (str): Path to the directory containing the data.
        batch_size (int): Batch size for training and validation dataloaders.
        data_fraction (float): Fraction of the dataset to use.
        syn_data (bool): Flag indicating whether to use synthetic data.
        copy_paste (bool): Flag indicating whether to use copy-paste augmentation.
        num_workers (int): Number of workers for data loading.
        annFile_src (str): Path to the annotation file for source dataset.
        annFile_tgt (str): Path to the annotation file for target dataset.
        root (str): Root directory for the dataset.
        img_sz (int): Size of the input images.

    Attributes:
        data_dir (str): Path to the directory containing the data.
        batch_size (int): Batch size for training and validation dataloaders.
        data_fraction (float): Fraction of the dataset to use.
        val_transform (torchvision.transforms.Compose): Transformations applied to validation data.
        syn_data (bool): Flag indicating whether to use synthetic data.
        copy_paste (bool): Flag indicating whether to use copy-paste augmentation.
        num_workers (int): Number of workers for data loading.
        instance_retriever (InstanceRetriever): Instance retriever for synthetic data.
        train (Dataset): Training dataset.
        val (Dataset): Validation dataset.

    """

    def __init__(
        self,
        data_dir: str = "data/data-llm",
        batch_size: int = 32,
        data_fraction: float = 1.0,
        syn_data: bool = False,
        copy_paste: bool = False,
        num_workers: int = 0,
        annFile_src: str = "subsets/instances_train2017_subset_0.1.json",
        annFile_tgt: str = "subsets/instances_train2017_subset_0.1.json",
        root: str = "data/data-llm",
        img_sz=1024,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_fraction = data_fraction
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.syn_data = syn_data
        self.copy_paste = copy_paste
        self.num_workers = num_workers
        self.img_sz = img_sz
        self.annFile_src = annFile_src
        self.annFile_tgt = annFile_tgt

    def setup(self, stage=None):
        if stage == "fit":
            if self.copy_paste:
                root_dir = os.environ.get("COCO_DATA_DIR_TRAIN")
                transform = T.Compose(
                    [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
                )
                target = CocoDetection(
                    root=root_dir, annFile=self.annFile_tgt, transform=transform
                )
                target = datasets.wrap_dataset_for_transforms_v2(
                    target, target_keys=["boxes", "labels", "masks"]
                )
                source = SubsampleCOCO(
                    root_dir, self.annFile_src, transform=transform, sample_n=1
                )
                self.train = PasteDataset2(source, target, self.img_sz)
            else:
                lsj = T.Compose(
                    [
                        T.ScaleJitter((self.img_sz, self.img_sz), (0.1, 2.0)),
                        adaptive_pad(self.img_sz, self.img_sz),
                        T.RandomCrop([self.img_sz, self.img_sz]),
                        T.SanitizeBoundingBoxes(),
                    ]
                )
                self.train = CoCoDet(
                    os.environ.get("COCO_DATA_DIR_TRAIN"),
                    self.annFile_tgt,
                    transform=lsj,
                )

            transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
            self.val = CoCoDet(
                os.environ.get("COCO_DATA_DIR_VAL"),
                os.environ.get("COCO_VAL_ANN"),
                transform=transform,
            )

        elif stage == "test":
            transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
            self.test = CoCoDet(
                os.environ.get("COCO_DATA_DIR_TRAIN"),
                self.annFile_tgt,
                transform=transform,
            )

    def train_dataloader(self):
        """
        Get the dataloader for training.

        Returns:
            torch.utils.data.DataLoader: The training dataloader.

        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Get the dataloader for validation.

        Returns:
            torch.utils.data.DataLoader: The validation dataloader.

        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    def test_dataloader(self):
        """
        Get the dataloader for testing.

        Returns:
            torch.utils.data.DataLoader: The testing dataloader.

        """
        return DataLoader(
            self.test,
            batch_size=2,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = transposed_data[0]
        self.tgt = transposed_data[1]

    # custom memory pinning method on custom type
    def pin_memory(self):
        # self.inp = self.inp.pin_memory()
        for i in range(len(self.tgt)):
            self.tgt[i]["masks"] = self.tgt[i]["masks"].pin_memory()
            self.tgt[i]["boxes"] = self.tgt[i]["boxes"].pin_memory()
            self.tgt[i]["labels"] = self.tgt[i]["labels"].pin_memory()
        inp = []
        for i in range(len(self.inp)):
            # print(type(self.inp[i]))
            # self.inp[i] = self.inp[i].pin_memory()
            inp.append(self.inp[i].pin_memory())
        print("pinned")

        return inp, self.tgt


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


def data_subset(dataset, fraction):
    # Calculate half of the length of the dataset
    num_samples = int(len(dataset) * fraction)
    # Create a range of indices from 0 to num_samples-1
    indices = range(num_samples)
    return Subset(dataset, indices)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # seed everything for reproducibility
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    dm = DM(num_workers=1, batch_size=32, copy_paste=False)
    dm.setup("fit")
    dl = dm.train_dataloader()
    for batch in dl:
        batch = pastebatched(batch, batch)
        for img, trg in zip(*batch):
            fig = vis_image_mask(img, trg)
            #breakpoint()
            name = id_generator()
            # make background black
            #plt.gca().set_facecolor("black")
            plt.savefig(f"plots/{name}.png", bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
        break
