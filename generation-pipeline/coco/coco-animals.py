import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

coco_annotation_file = '/path2anno'
# Load the COCO annotation file
coco = COCO(coco_annotation_file)
# Get all categories
categories = coco.loadCats(coco.getCatIds())
# Filter categories that belong to the "animal" supercategory
animal_categories = [cat for cat in categories if cat['supercategory'] == 'animal']
print(animal_categories)
animal_categories.remove({'supercategory': 'animal', 'id': 16, 'name': 'bird'})
# Get the category IDs of animal classes
animal_category_ids = [cat['id'] for cat in animal_categories]
# Initialize a dictionary to store the class frequencies
class_frequencies = {cat['name']: 0 for cat in animal_categories}


# Iterate over all images
for img_id in tqdm(coco.getImgIds()):
    # Get the annotations for the current image
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=animal_category_ids)
    anns = coco.loadAnns(ann_ids)
    
    # Count the occurrences of each animal class in the current image
    for ann in anns:
        class_name = coco.loadCats(ann['category_id'])[0]['name']
        class_frequencies[class_name] += 1

# Count the number of images containing animals
num_images_with_animals = len(set(ann['image_id'] for ann in coco.loadAnns(coco.getAnnIds(catIds=animal_category_ids))))
print(f"Number of images containing animals: {num_images_with_animals}")
# Print the class frequencies
for class_name, frequency in class_frequencies.items():
    print(f"{class_name}: {frequency}")

# Plot the class frequency distribution
plt.figure(figsize=(12, 6))
plt.bar(class_frequencies.keys(), class_frequencies.values())
plt.xlabel('Animal Class')
plt.ylabel('Frequency')
plt.title('Class Frequency Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


