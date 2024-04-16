import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import json

import matplotlib.pyplot as plt

# Load the COCO annotation file
coco_annotation_file = '/path2anno'
coco = COCO(coco_annotation_file)

# Get the category IDs and names
cat_ids = coco.getCatIds()
cat_names = [coco.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]

# Initialize a dictionary to store the class instance counts
class_counts = {cat_name: 0 for cat_name in cat_names}

# Iterate over all the images in the dataset
img_ids = coco.getImgIds()
for img_id in tqdm(img_ids, desc='Calculating class instance distribution'):
    # Get the annotations for the current image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    # Count the instances of each class in the current image
    for ann in anns:
        cat_id = ann['category_id']
        cat_name = coco.loadCats(cat_id)[0]['name']
        class_counts[cat_name] += 1

# sort class by freqs before plotting and saving
sorted_class_counts = {k: v for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)}

# Plot histogram
plt.figure(figsize=(12, 6))  # Set the figure size to be larger
plt.bar(sorted_class_counts.keys(), sorted_class_counts.values())
plt.xticks(rotation=90)
plt.xlabel('Class')
plt.ylabel('Instance Count')
plt.title('Class Instance Distribution')
plt.tight_layout()
plt.savefig('/generation-pipeline/class_freqs.png')

# Save histogram as JSON
with open('/generation-pipeline/class_counts.json', 'w') as f:
    json.dump(sorted_class_counts, f)
    
