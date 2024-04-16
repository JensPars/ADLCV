from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from simple_copy_paste.simple_copy_paste import copy_paste_class, CopyPaste
from simple_copy_paste.coco import CocoDetectionCP 
from simple_copy_paste.coco import CustomCocoDetection
import albumentations as A


class CocoDataModule(LightningDataModule):
    def __init__(self, train_dir, train_file, val_dir, val_file, transforms, batch_size):
        super().__init__()
        self.train_dataset = CustomCocoDetection(train_dir, train_file, transforms=transforms)
        self.val_dataset = CustomCocoDetection(val_dir, val_file, transforms=transforms)
        #self.train_dataset = CocoDetectionCP(train_dir, train_file, transforms=transforms)

        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,collate_fn=self.train_dataset.collate_fn, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, collate_fn=self.val_dataset.collate_fn, persistent_workers=True) 
