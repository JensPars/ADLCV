import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

from torchvision.datasets import CocoDetection
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


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
    A.Resize(256, 256),
    CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
    AT.ToTensorV2(),

],bbox_params=A.BboxParams(format="coco"))

#train_dataset = CocoDetectionCP(root='/work3/s194633/train2017', annFile='bear_subset_annotations_train.json', transforms=transform)

train_dataset = CocoDetection(root='/work3/s194633/train2017', annFile='bear_subset_annotations_train.json', transform=val_transform)
train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset, target_keys=["boxes", "labels", "masks"])
val_dataset = CocoDetection(root='/work3/s194633/val2017', annFile='bear_subset_annotations_val.json', transform=val_transform)
val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset, target_keys=["boxes", "labels", "masks"])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn_v2(weights=None,num_classes=2)
model.to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]

# Define optimizer
optimizer = torch.optim.AdamW(params, lr=2e-4)
# Define the ReduceLROnPlateau learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

num_epochs = 30
best_val_loss = float('inf')

# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, targets in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        train_loss += losses.item()

    train_loss /= len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {train_loss:.4f}")

    # Validation
    val_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
            model.train()  # Ensure model is in train mode for consistency
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}")

    # Adjust learning rate based on validation loss
    lr_scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("Saving new best model")
        torch.save(model.state_dict(), '/work3/s194633/plain_best_maskrcnn_bear.pth')

print(f"Best Validation Loss: {best_val_loss:.4f}")