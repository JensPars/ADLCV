import os
import torch
import cv2
from torchvision.datasets import CocoDetection
from simple_copy_paste.simple_copy_paste import copy_paste_class
from torchvision import tv_tensors
import numpy as np

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
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

@copy_paste_class
class CocoDetectionCP(CocoDetection):
    def __init__(
        self,
        root,
        annFile,
        transforms
    ):
        super(CocoDetectionCP, self).__init__(
            root, annFile, None, None, transforms
        )

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids
    
    def collate_fn(_, batch):
        return tuple(zip(*batch))

    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #convert all of the target segmentations to masks
        #bboxes are expected to be (y1, x1, y2, x2, category_id)
        masks = []
        bboxes = []
        for ix, obj in enumerate(target):
            masks.append(self.coco.annToMask(obj))
            bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])

        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }
        
        return self.transforms(**output)
    
class CustomCocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file, transforms)
        self.img_folder = img_folder

    @staticmethod
    def collate_fn(batch):
        images = list(image for image, _ in batch)  
        targets = list(target for _, target in batch) 
        return images, targets

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        target = self.coco.loadAnns(ann_ids)

        # Process target to match your desired format
        output_target = {}
        output_target['boxes'] = []
        output_target['labels'] = []
        output_target['masks'] = []

        for obj in target:
            # Convert boxes to x1, y1, x2, y2
            x, y, w, h = obj['bbox']
            x2 = x + w
            y2 = y + h
            output_target['boxes'].append(torch.tensor([x, y, x2, y2]).float())

            output_target['labels'].append(obj['category_id'])

            # Assuming masks are RLE encoded
            mask = self.coco.annToMask(obj) 
            output_target['masks'].append(mask) 


        path = self.coco.loadImgs(image_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.img_folder, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).float().permute(2, 0, 1)

        if self.transforms is not None:
            image, output_target = self.transforms(image, output_target)

        # Convert to tensors
        output_target['boxes'] = torch.tensor(np.array(output_target['boxes'])).float()  # Ensure float for bounding boxes
        output_target['labels'] = torch.tensor(output_target['labels']).long()
        output_target['masks'] = torch.tensor(np.array(output_target['masks']))

        return image, output_target