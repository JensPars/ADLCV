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
val_dataset = CocoDetection(root='/work3/s194649/train2017', annFile='bear_subset_annotations_val.json', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=False, weights=None)
model.to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #breakpoint()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {losses.item():.4f}")
    
    lr_scheduler.step()
    
    # Validation
    model.eval()
    #with torch.no_grad():
        #for images, targets in val_loader:
            #images = [image.to(device) for image in images]
            #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            #outputs = model(images)
            
            ## Compute validation metrics
            
    
    #print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {losses.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'maskrcnn_coco.pth')