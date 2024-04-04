import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import numpy as np
from pycocotools.coco import annToRLE
from pycocotools.mask import encode
def get_instance_masks(coco_annotation_file, class_name, output_size=(512, 512), T=1024):
    # Load the COCO annotation file
    coco = COCO(coco_annotation_file)

    # Get the category ID for the given class name
    category_id = coco.getCatIds(catNms=[class_name])[0]

    # Get the image IDs that contain the specified class
    image_ids = coco.getImgIds(catIds=[category_id])

    masks = []
    for image_id in tqdm(image_ids, desc='Processing images'):
        # Get the annotations for the current image
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=[category_id]))

        # Create a blank mask for the current image
        image_info = coco.loadImgs([image_id])[0]
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        # Iterate over the annotations and draw the segmentation masks
        for annotation in annotations:
            segmentation = annotation['segmentation']
            mask = coco.annToMask(annotation)

            # Crop the mask to the bounding box of the instance
            bbox = annotation['bbox']
            x, y, w, h = [int(coord) for coord in bbox]
            if h*w>T:
                instance_mask = mask[y:y+h, x:x+w]
                # Resize the instance mask to the desired output size
                instance_mask = Image.fromarray(instance_mask)
                instance_mask = instance_mask.resize(output_size, resample=Image.NEAREST)
                instance_mask = np.array(instance_mask)

                yield instance_mask


# Example usage
coco_annotation_file = '/work3/s194649/annotations/instances_train2014.json'
class_name = 'person'  # Replace with your desired class name

instance_masks = get_instance_masks(coco_annotation_file, class_name, T=128**2)
masks = [mask for _,mask in zip(range(100),instance_masks)]
masks_tensor = np.stack(masks)
# Create a new COCO dataset for the masks


for i, mask in enumerate(masks):
    # Create a new image entry
    image_id = i + 1
    # Create a new annotation entry
    rle = encode(np.asfortranarray((mask)))
    annotation = {
        "caption": "A person in a leather coat",
        "width": 512,
        "height": 512,
        "annos": [{"mask": rle,
        "category_name": "person",
        "caption": "A person in a leather coat"}]
    }
    break
    
