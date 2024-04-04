import os
import random
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO

# Set the root directory and annotation file path
root = "work3/s194649/coco"
anno = "/zhome/ca/9/146686/ADLCV/coco_subset_annotations.json"

# Create the CocoDetection dataset
dataset = CocoDetection(root, anno)

# Create a COCO object for accessing annotations
coco = dataset.coco

# Get the category IDs and their corresponding names
cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)
cat_names = [cat['name'] for cat in cats]

# Select 100 random images from the dataset
num_images = 100
image_ids = random.sample(coco.getImgIds(), num_images)

# Create a grid of subplots for displaying the images
rows = 10
cols = 10
fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

# Iterate over the selected images and plot them with annotations
for i, img_id in enumerate(image_ids):
    # Load the image
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(root, img_info['file_name'])
    img = plt.imread(img_path)

    # Get the annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Plot the image and annotations
    ax = axs[i // cols, i % cols]
    ax.imshow(img)
    ax.axis('off')

    # Plot the bounding boxes and category names
    for ann in anns:
        bbox = ann['bbox']
        cat_id = ann['category_id']
        cat_name = cat_names[cat_ids.index(cat_id)]
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, cat_name, fontsize=8, color='r', verticalalignment='top')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()