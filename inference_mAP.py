from generation_pipeline.setupHF_cache import *
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchmetrics.detection import MeanAveragePrecision
import tqdm
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from transformers import DetrImageProcessor, DetrForObjectDetection

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm", do_rescale=False)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to("cpu")

# Load COCO dataset
val_dataset = CocoDetection(root='/work3/s194649/val2017', annFile='/work3/s194649/annotations/instances_val2017.json', transform=transform)
val_dataset = datasets.wrap_dataset_for_transforms_v2(val_dataset, target_keys=["boxes", "labels", "masks"])
batch_size = 16
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda s:tuple(zip(*s)))
with open('cats.json', 'w') as f:
    json.dump(val_dataset.coco.cats, f)
#model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
#model.to(device)
#model.eval()

# Initialize MeanAveragePrecision metric
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

# Iterate over the validation dataset
for (images, targets) in tqdm.tqdm((val_loader)):
    #images = [image.to(device) for image in images]
    image_sizes = [tuple(img.shape[-2:]) for img in images]
    images = processor(images, return_tensors="pt").to("cuda")
    
    targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, int)} for t in targets]

    with torch.no_grad():
        outputs = model(**images)
    
    outs = processor.post_process_object_detection(outputs, target_sizes=image_sizes, threshold=0)
        
    #preds = []
    #for output in outputs:
    #    pred = {
    #        #'masks': output['masks'].squeeze(1).bool(),#.cpu().bool(),
    #        'boxes': output['boxes'],
    #        'scores': output['scores'],
    #        'labels': output['labels']
    #    }
    #    preds.append(pred)
    
    #gts = []
    #try:
    #    for output in targets:
    #        pred = {
    #            #'masks': output['masks'].bool(),#.cpu().bool(),
    #            'boxes': output['boxes'],
    #            'labels': output['labels']
    #        }
    #        gts.append(pred)
    #        metric.update(preds, gts)
    
    try:
        metric.update(outs, targets)
    except:
        print("Failed")
    #if i == 100:
    #    break

# Compute mAP
results = metric.compute()
print("mAP Results:")
print(results)
