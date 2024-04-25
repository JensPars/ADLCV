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
from argparse import ArgumentParser
import wandb
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import Subset


def data_subset(dataset,fraction):
    # Calculate half of the length of the dataset
    num_samples = int(len(dataset)*fraction)
    # Create a range of indices from 0 to num_samples-1
    indices = range(num_samples)
    return Subset(dataset, indices)


# Argument parsing
parser = ArgumentParser(description="Choose the COCO dataset version for training.")
parser.add_argument("--copy_paste", help="Use COCOCP dataset for training.")
parser.add_argument("--data_fraction", help="portion of data to use for training", default=1., type=float)
args = parser.parse_args()

wandb.init(project = "copy-paste-project", name=f"mask-rcnn-{'copy-paste' if args.copy_paste=='True' else 'plain'}-{str(args.data_fraction)}-data")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Define transformations
val_transform = transforms.Compose([
    transforms.ToTensor(),
    #A.Resize(512, 512),
])

transform = A.Compose([
    A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
    A.PadIfNeeded(512, 512, border_mode=0), #constant 0 border
    A.Resize(512, 512),
    CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
    AT.ToTensorV2(),

],bbox_params=A.BboxParams(format="coco"))
if args.copy_paste == "True":
    train_dataset = CocoDetectionCP(root='/work3/s194633/train2017', annFile='car_boat_bus_train.json', transforms=transform)
else:
    train_dataset = CocoDetection(root='/work3/s194633/train2017', annFile='car_boat_bus_train.json', transform=val_transform)
    train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset, target_keys=["boxes", "labels", "masks"])

train_dataset = data_subset(train_dataset,args.data_fraction)

val_dataset = CocoDetection(root='/work3/s194633/val2017', annFile='car_boat_bus_val.json', transform=val_transform)
val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset, target_keys=["boxes", "labels", "masks"])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))


# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn_v2(weights=None, num_classes=4)
model.to(device)
wandb.watch(model, log='all')

# Define optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

num_epochs = 50
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
    wandb.log({"train_loss": train_loss})

    # Validation
    val_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}")
    wandb.log({"val_loss": val_loss})

    # Adjust learning rate based on validation loss
    lr_scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"Saving new best model at Epoch {epoch+1}")
        model_path = f'/work3/s194633/{"copy_paste" if args.copy_paste=="True" else "plain"}_best_maskrcnn_car_boat_bus_train_{str(args.data_fraction)}_data.pth'
        torch.save(model.state_dict(), model_path)

print(f"Best Validation Loss: {best_val_loss:.4f}")

# Define transformations for test data
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = CocoDetection(root='/work3/s194633/val2017', annFile="car_boat_bus_val.json", transform=test_transform)

# Wrap dataset
test_dataset = datasets.wrap_dataset_for_transforms_v2(test_dataset, target_keys=["boxes", "labels", "masks"])

# Create DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

# Load the best model
model = maskrcnn_resnet50_fpn_v2(weights=None, num_classes=4)
model.load_state_dict(torch.load(f'/work3/s194633/{"copy_paste" if args.copy_paste=="True" else "plain"}_best_maskrcnn_car_boat_bus_train_{str(args.data_fraction)}_data.pth'))
model.to(device)
model.eval()

# Initialize metric
map_metric = MeanAveragePrecision()
# Disable gradient computation for testing
with torch.no_grad():
    for images, targets in test_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        # Update the metric
        map_metric.update(outputs, targets)

# Finalize the metric computation
map_dict = map_metric.compute()
print(map_dict)
map_val = map_dict['map']

# Log Mean Average Precision to W&B
wandb.log({"Test mAP": map_val})

# Print out Mean Average Precision
print(f"Test mAP: {map_val:.4f}")

# Finish W&B run
wandb.finish()
wandb.finish()
