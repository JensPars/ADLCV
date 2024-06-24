import os
import random
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

ann_file = '/work3/s194649/annotations/instances_train2017.json'
root_dir = '/work3/s194649/train2017'
coco = COCO(ann_file)
# Get the image ids with annotations
img_ids_with_anns = []
annos = coco.dataset['annotations']
for id in coco.getCatIds():
    for el in coco.getImgIds(catIds=id):
        if el not in img_ids_with_anns:
            img_ids_with_anns.append(el)


# Create a new annotation dictionary
new_annotations = {'images': [], 'annotations': annos, 'categories': coco.dataset['categories']}

# Iterate over the images with annotations
for img_id in img_ids_with_anns:
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    # Check if there are any annotations for the image
    if anns:
        print("hej")
        new_annotations['images'].append(img_info)
        new_annotations['annotations'].extend(anns)

# Save the new annotation file
import json
with open('cleaned_train.json', 'w') as f:
    json.dump(new_annotations, f)