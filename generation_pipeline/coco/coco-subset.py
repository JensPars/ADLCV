import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Load the COCO annotation file
coco_annotation_file = '/work3/s194649/annotations/instances_train2017.json'
coco = COCO(coco_annotation_file)

# Create a new dictionary to store the subset annotations
subset_annotations = {
                        'info': coco.dataset['info'],
                        'licenses': coco.dataset['licenses'],
                        'categories': coco.dataset['categories'],
                        'images': [],
                        'annotations': []
                     }

# Get image ids that have annotations
img_ids = [img_id for img_id in coco.getImgIds() if len(coco.getAnnIds(imgIds=img_id)) > 0]
# shuffle the image ids
np.random.shuffle(img_ids)
ps = [0.1, 0.25, 0.5, 0.75, 1.0]
# generate subsets
for p in ps:
    imgs_ids_subset = img_ids[:int(p * len(img_ids))]
    print(f"Subset size: {len(imgs_ids_subset)}")
    
    # Get the image and annotation information for the subset
    imgs_subset = coco.loadImgs(imgs_ids_subset)
    annIds = coco.getAnnIds(imgIds=imgs_ids_subset)
    anns_subset = coco.loadAnns(annIds)
    
    # Create a new annotation file for the subset
    subset_annotations['images'] = imgs_subset
    subset_annotations['annotations'] = anns_subset
    
    # Save the subset annotation file
    with open(f'subsets/instances_train2017_subset_{p}.json', 'w') as f:
        json.dump(subset_annotations, f)