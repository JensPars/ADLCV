import os
import random
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO

# Set the root directory and annotation file path
root = "/work3/s194649/train2017"
anno = "coco_subset_annotations.json"

coco = COCO("coco_subset_annotations.json")

# Get the category IDs and their corresponding names
cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)
cat_names = [cat['name'] for cat in cats]

# Select 100 random images from the dataset
num_images = 10
image_ids = random.sample(coco.getImgIds(), num_images)

for img_id in image_ids:
    # Load the image
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(root, img_info['file_name'])
    img = plt.imread(img_path)
    # Get the annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    # Plot the image and annotations
    plt.imshow(img)
    plt.axis('off')
    coco.showAnns(anns)
    # Adjust the spacing between subplots
    plt.tight_layout()  
    # Display the plot
    plt.savefig(f"data/example{img_id}.png")
    plt.close()


