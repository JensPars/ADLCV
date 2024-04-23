import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

from torchvision.datasets import CocoDetection
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


import albumentations.pytorch as AT
import albumentations as A
import matplotlib.pyplot as plt
from simple_copy_paste.coco import CocoDetectionCP
from simple_copy_paste.simple_copy_paste import CopyPaste
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Define transformations
val_transform = transforms.Compose([
    transforms.ToTensor(),
])

transform = A.Compose([
    A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
    A.PadIfNeeded(256, 256, border_mode=0), #constant 0 border
    A.RandomCrop(256, 256),
    CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
    AT.ToTensorV2(),

],bbox_params=A.BboxParams(format="coco"))

train_dataset = CocoDetectionCP(root='data/val2017', annFile='bear_subset_annotations_val.json', transforms=transform)

#plot the first ten images with masks and bounding boxes
# print(train_dataset[0][0].max())
# fig, ax = plt.subplots(2, 5, figsize=(20, 10))
# for i in range(10):
#     image, target = train_dataset[i]
#     image = (image).permute(1, 2, 0).numpy()
#     ax[i // 5, i % 5].imshow(image)
#     for box, mask in zip(target['boxes'], target['masks']):
#         box = box.int().cpu().numpy()
#         # mask = mask.mul(255).byte().cpu().numpy()
#         ax[i // 5, i % 5].add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], edgecolor='red', fill=False))
#         ax[i // 5, i % 5].imshow(mask, alpha=0.1)
# plt.show()

#train_dataset = CocoDetection(root='data/val2017', annFile='bear_subset_annotations_val.json', transform=val_transform)
#train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset, target_keys=["boxes", "labels", "masks"])
val_dataset = CocoDetection(root='data/val2017', annFile='bear_subset_annotations_val.json', transform=val_transform)
val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset, target_keys=["boxes", "labels", "masks"])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(weights=None,num_classes=2)
model.to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]

import numpy as np

# Define optimizer
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler parameters
initial_lr = 0.005
warmup_start_lr = 0.001
warmup_epochs = 5

# Function for calculating the learning rate
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        # Linear warm-up
        lr = warmup_start_lr + (initial_lr - warmup_start_lr) * (epoch / warmup_epochs)
    else:
        # Apply step decay after warm-up
        lr_decay_step = 3  # Step decay every 3 epochs
        lr_decay_factor = 0.1
        lr = initial_lr * (lr_decay_factor ** ((epoch - warmup_epochs) // lr_decay_step))
    return lr / initial_lr

# Define the scheduler
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


num_epochs = 50
best_val_loss = float('inf')

# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        print(losses)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        train_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {train_loss:.4f}")

    lr_scheduler.step()

    # Validation
    val_loss = 0
    map_metric = MeanAveragePrecision()
    with torch.no_grad():
        for images, targets in val_loader:
            model.train()
            images = [image.to(device) for image in images]
            targets = [target for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses

            model.eval()
            outputs = model(images)
            # Calculate mAP using torchmetrics
            map_metric.update(outputs, targets)
            map_dict = map_metric.compute()
            map_val = map_dict['map']
            
    
    print(f"Epoch [{epoch+1}/{num_epochs}] mAP: {map_val:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("Saving new best model")
        torch.save(model.state_dict(), 'best_maskrcnn_coco.pth')

# Optionally, print out best validation loss at the end
print(f"Best Validation Loss: {best_val_loss:.4f}")