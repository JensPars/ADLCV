import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import datasets

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for test data
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = CocoDetection(root='/work3/s194633/val2017', annFile='bear_subset_annotations_val.json', transform=test_transform)

# Wrap dataset [if needed, depending on library version if target keys are required for metric computation]
test_dataset = datasets.wrap_dataset_for_transforms_v2(test_dataset, target_keys=["boxes", "labels", "masks"])

# Create DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

# Load the best model
model = maskrcnn_resnet50_fpn(weights=None, num_classes=2)
model.load_state_dict(torch.load('/work3/s194633/simple_copy_paste_best_maskrcnn_bear.pth'))
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

# Print out Mean Average Precision
print(f"Test mAP: {map_val:.4f}")