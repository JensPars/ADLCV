import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

classes = {'bus':0, 'car':0, 'boat':0}

# Load the COCO annotation file
coco_annotation_file = 'data/annotations/instances_train2017.json'
coco = COCO(coco_annotation_file)

# Create a new dictionary to store the subset annotations
subset_annotations = {
                        'info': coco.dataset['info'],
                        'licenses': coco.dataset['licenses'],
                        'categories': [],
                        'images': [],
                        'annotations': []
                     }

# Create a mapping of old category IDs to new category IDs
category_id_map = {}
for i, (class_name, _) in enumerate(classes.items(), start=1):
    category = next(c for c in coco.dataset['categories'] if c['name'] == class_name)
    print(class_name, category['id'])
    category_id_map[category['id']] = i
    subset_annotations['categories'].append({
        'id': i,
        'name': class_name,
        'supercategory': category['supercategory']
    })

# # Filter annotations and images
# image_ids = set()
# for ann in tqdm(coco.dataset['annotations']):
#     if ann['category_id'] in category_id_map:
#         ann['category_id'] = category_id_map[ann['category_id']]
#         #ann.pop('segmentation', None)  # Remove segmentation (masks)
#         subset_annotations['annotations'].append(ann)
#         image_ids.add(ann['image_id'])

# # Filter images
# for image in coco.dataset['images']:
#     if image['id'] in image_ids:
#         subset_annotations['images'].append(image)

# print(len(subset_annotations['images']))

# # Save the subset annotations as a nicely formatted JSON file
# output_file = 'car_boat_bus_train.json'
# with open(output_file, 'w') as f:
#     json.dump(subset_annotations, f, indent=4)

# print(f"Subset annotations saved to {output_file}")