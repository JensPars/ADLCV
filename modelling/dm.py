from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from simple_copy_paste.simple_copy_paste import copy_paste_class, CopyPaste
from simple_copy_paste.coco import CocoDetectionCP 
from simple_copy_paste.coco import CustomCocoDetection, CustomCocoSegmentation
import albumentations as A


class CocoDataModule(LightningDataModule):
    def __init__(self, train_dir, train_file, val_dir, val_file, transforms_im, transforms_mask, batch_size):
        super().__init__()
        self.train_dataset = CustomCocoSegmentation(train_dir, train_file, transforms_im=transforms_im, transforms_mask=transforms_mask)
        self.val_dataset = CustomCocoSegmentation(val_dir, val_file, transforms_im=transforms_im, transforms_mask=transforms_mask)
        #self.train_dataset = CocoDetectionCP(train_dir, train_file, transforms=transforms)

        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1,persistent_workers=True) 
