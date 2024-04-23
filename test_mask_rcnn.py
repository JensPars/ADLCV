import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import datasets
from torchvision.transforms import transforms
import wandb
from argparse import ArgumentParser


parser = ArgumentParser(description="Choose the COCO dataset version for training.")
parser.add_argument("--copy_paste", action="store_true", help="Use COCOCP dataset for training.", required=True)
args = parser.parse_args()
# Initialize Weights & Biases - continue from an existing run or start a new run
# If continuing, replace 'your-run-id' with your specific run id
wandb.init(project = "copy-paste-project", name=f"mask-rcnn-{'copy-paste' if args.copy_paste else 'plain'}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for test data
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = CocoDetection(root='/work3/s194633/val2017', annFile='bear_subset_annotations_val.json', transform=test_transform)

# Wrap dataset
test_dataset = datasets.wrap_dataset_for_transforms_v2(test_dataset, target_keys=["boxes", "labels", "masks"])

# Create DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

# Load the best model
model = maskrcnn_resnet50_fpn_v2(weights=None, num_classes=2)
model.load_state_dict(torch.load(f'/work3/s194633/{"copy_paste" if args.copy_paste else "plain"}_best_maskrcnn_bear.pth'))
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