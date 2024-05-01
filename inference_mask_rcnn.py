import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load COCO dataset
train_dataset = CocoDetection(root='/work3/s194649/train2017', annFile='bear_subset_annotations_train.json', transform=transform)
train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset, target_keys=["boxes", "labels", "masks"])
val_dataset = CocoDetection(root='/work3/s194649/val2017', annFile='/work3/s194649/annotations/instances_val2017.json', transform=transform)
coco_lbls = val_dataset.coco.cats
val_dataset = CocoDetection(root='/work3/s194649/val2017', annFile='bear_subset_annotations_val.json', transform=transform)

batch_size = 8
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda s:zip(*s))

model = maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

import matplotlib.pyplot as plt
import numpy as np

# Iterate over the validation dataset
for images, targets in val_loader:
    images = [image.to(device) for image in images]
    
    with torch.no_grad():
        outputs = model(images)
    
    # Plot the images and predicted masks
    for i in range(len(images)):
        image = images[i].permute(1, 2, 0)#.cpu().numpy()
        boxes = outputs[i]['boxes']#.cpu().numpy()
        labels = outputs[i]['labels']#.cpu().numpy()
        scores = outputs[i]['scores']#.cpu().numpy()
        masks = outputs[i]['masks'].squeeze(1).cpu().numpy()
        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Plot the predicted bounding boxes and masks
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            if score > 0.5:  # Filter predictions based on confidence threshold
                # Draw bounding box
                xmin, ymin, xmax, ymax = box.astype(int)
                plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                                  fill=False, edgecolor='r', linewidth=2))
                
                # Draw mask
                mask = mask > 0.5  # Threshold the mask
                plt.imshow(np.ma.masked_where(~mask, mask), alpha=0.5, cmap='viridis')
                # Add label and score
                label_name = coco_lbls[label]['name']
                plt.gca().text(xmin, ymin, f"{label_name}: {score:.2f}", color='white',
                               bbox=dict(facecolor='red', alpha=0.5))
        
        plt.axis('off')
        plt.savefig(f'data/preds/{i}i.png')