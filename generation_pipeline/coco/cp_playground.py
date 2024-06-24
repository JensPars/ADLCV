import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy
from skimage.filters import gaussian
import torch
from torchvision import tv_tensors
from PIL import Image

class CPdataset():
    def __init__(self, instance_retriever, layering:str, n_instances:int, min_visible_pct:float=1.0, min_visible_pixels:int = 32):
        self.instance_retriever = instance_retriever
        self.min_visible_pct = min_visible_pct
        self.n_instances = n_instances
        self.layering = layering # TODO: Implement the layering
        self.min_visible_pixels = min_visible_pixels

    def __call__(self, image, masks, bboxes, labels):
        return self.copy_paste(image, masks, bboxes, labels)

    def copy_paste(self, image, masks, bboxes, labels):

        # Background Image
        background = Image.fromarray(image)
        # Collect new instances to paste onto the background
        instances = self.instance_retriever.get_instances(self.n_instances)
        new_masks = []
        new_bboxes = []
        #instances = self._apply_layering(instances, masks, bboxes, labels)
        # Paste the new instances onto the background
        for instance in instances:
            out = self._paste_single_instance(instance['image'], 
                                            instance['masks'][0], 
                                            background, 
                                            min_visible_pct=self.min_visible_pct)
            background, new_instance_mask, new_instance_bbox = out
            new_instance_bbox = new_instance_bbox # Add label to the bounding box
            new_masks.append(new_instance_mask)
            new_bboxes.append(new_instance_bbox)
        # Collect all bounding boxes and masks
        bboxes.extend(new_bboxes)
        masks.extend(new_masks)
        labels.extend([instances[i]['labels'] for i in range(len(instances))])

        # Adjust the masks and bounding boxes to make sure they are consistent and that masks do not overlap
        adjusted_masks, adjusted_bboxes, adjusted_labels = self._adjust_masks_and_bboxes(masks, bboxes, labels)
        if isinstance(background, Image.Image): 
            background = np.array(background)
        if labels != []:
            background = self._blend_masks2background(image, background, adjusted_masks, sigma = 1.0)

        return background, {"masks": adjusted_masks, "boxes": adjusted_bboxes, "labels": adjusted_labels}

    def _blend_masks2background(self, original_image, pasted_image, mask, sigma:float):
        """
        Blend the pasted image with the background image.
        """
        # Ensure mask is a single channel and in the correct format
        if isinstance(mask, list):
            mask = np.stack(mask, axis=0)
        if len(mask.shape) > 2:
            mask = mask.sum(axis=0)

        # Optionally apply Gaussian blur to the mask for smooth blending
        mask = gaussian(mask, sigma = sigma, preserve_range=True)
        
        # Perform blending
        mask = mask[..., None] # W x H x 1
        blended_area = pasted_image * mask + original_image * (1 - mask)
        blended_area = blended_area.astype(pasted_image.dtype)
        return blended_area

    def _adjust_masks_and_bboxes(self, masks, bboxes, labels):
        # Stack masks into a 3D numpy array
        masks = np.stack(masks, axis=0)

        # Get the index of the highest priority mask for each pixel
        highest_priority_mask = np.argmax(masks[::-1], axis=0)
        highest_priority_mask = masks.shape[0] - 1 - highest_priority_mask
        
        # Create an array where each mask is compared to the highest priority mask
        final_masks = np.zeros_like(masks)
        for i in range(masks.shape[0]):
            final_masks[i] = (highest_priority_mask == i) * masks[i]

        # Convert array back to the list of masks
        new_masks = list(final_masks)
        
        # Compute visibility of pixels per mask
        visible_pixels_per_mask = final_masks.sum(axis=(1, 2))

        # Adjust the bounding boxes and remove masks with too few key points visible
        adjusted_bboxes = []
        adjusted_masks = []
        adjusted_labels = []
        
        for i in range(len(visible_pixels_per_mask)):
            if visible_pixels_per_mask[i] > self.min_visible_pixels:
                x1, y1, x2, y2 = self.extract_bbox(new_masks[i])
                if (x2-x1 > 0) and (y2-y1 > 0): # validate bounding box
                    adjusted_masks.append(new_masks[i])
                    adjusted_bboxes.append([x1, y1, x2, y2])
                    adjusted_labels.append(labels[i])
        
        return adjusted_masks, adjusted_bboxes, adjusted_labels
    
    def extract_bbox(self, mask):
        """
        Extract the bounding box from a segmentation mask.
        Output format is (x1,y1,w,h).
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        assert len(np.where(rows)[0]) > 0
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        assert ymax >= ymin and xmax >= xmin
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def _crop_image(self, image, bbox):
        """
        Crop an image given a bounding box.
        """
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    def _paste_single_instance(self, image, mask, background, min_visible_pct):
        """
        Paste a segmented part of an image such that at least K% of it remains within
        the background image boundaries.
        
        Args:
        image (PIL.Image): Input image of size n by n.
        mask (numpy.array): Segmentation mask, same dimensions as image.
        background (PIL.Image): Background image to paste onto.
        min_visible_pct (float): Minimum percentage of the image that should remain visible when pasted.
        
        Returns:
        PIL.Image: Modified background image with the pasted segment.
        numpy.array: Updated segmentation mask.
        tuple: Updated bounding box (xmin, ymin, xmax, ymax).
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(background, Image.Image):
            background = Image.fromarray(background)

        # Convert the segmentation mask to a PIL image and extract the region
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        segmented_part = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask_image)

        # Calculate the range for the random location
        min_x = int(-(1-min_visible_pct) * image.width)
        max_x = int(background.width - min_visible_pct * image.width)
        min_y = int(-(1-min_visible_pct) * image.height)
        max_y = int(background.height - min_visible_pct * image.height)

        random_x = random.randint(min_x, max_x)
        random_y = random.randint(min_y, max_y)

        # Paste the segmented part onto the background
        background.paste(segmented_part, (random_x, random_y), mask_image)

        # Update the mask and bounding box
        new_mask = np.zeros((background.height, background.width), dtype=np.uint8)

        # Determine coordinates on the background
        bx1 = max(random_x, 0)
        by1 = max(random_y, 0)
        bx2 = min(random_x + image.width, background.width)
        by2 = min(random_y + image.height, background.height)

        # Determine coordinates on the segmented part
        sx1 = max(-random_x, 0)
        sy1 = max(-random_y, 0)
        sx2 = sx1 + (bx2 - bx1)
        sy2 = sy1 + (by2 - by1)

        # Update new mask
        new_mask[by1:by2, bx1:bx2] = mask[sy1:sy2, sx1:sx2]

        new_bbox = (bx1, by1, bx2, by2)

        return background, new_mask, new_bbox


        

    



import torchvision
from torchvision.datasets import CocoDetection
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.v2 import ScaleJitter, Pad
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
# import to pil image
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from torchvision.transforms import v2
from torchvision import models, datasets, tv_tensors

#lsj = v2.RandomZoomOut(fill={tv_tensors.Image: (123/255, 117/255, 104/255), "others": 0}, side_range=(1, 2), p=1)


#v2.Pad((480, 640), fill=0, padding_mode="constant")

class adaptive_pad(v2.Transform):
    def __init__(self, h=1024, w=1024):
        self.h = h
        self.w = w
    
    def __call__(self, input):
        img = input[0]
        h, w = img.shape[-2:]
        pad_w = self.h - h
        pad_h = self.w - w
        pad = (pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2)
        # pad = (pad_w, pad_h, 0, 0)
        p_t = Pad(pad, fill=0, padding_mode="constant")
        return p_t(input)
        #return v2.functional.pad(img, pad, fill=0, padding_mode="constant"), v2.functional.pad(input[1], pad, fill=0, padding_mode="constant")

augs = v2.Compose([ScaleJitter((1024, 1024), (0.1, 1)), adaptive_pad(1024, 1024)]) #, adaptive_pad(1024, 1024) #A.PadIfNeeded(1024, 1024, border_mode=0)]) # adaptive_pad(1024, 1024)])      
ann_file = '/work3/s194649/annotations/instances_train2017.json'
root_dir = '/work3/s194649/train2017'
val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

dataset = CocoDetection(
                root=root_dir, annFile=ann_file, transform=val_transform
            )

dataset = datasets.wrap_dataset_for_transforms_v2(
                dataset, target_keys=["boxes", "labels", "masks"]
            )

scaler = A.RandomScale(
            scale_limit=(-0.9, 0), p=1
        )

obs = [augs(dataset[0]) for i in range(9)]
imgs = [o[0] for o in obs]
targets = [o[1] for o in obs]

obs2 = [augs(dataset[1]) for i in range(9)]
imgs2 = [o[0] for o in obs2]
targets2 = [o[1] for o in obs2]



def plot_images_with_boxes_and_masks(images, targets, num_images=4):
    fig, axs = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
    if num_images == 1:
        axs = [axs]  # Ensure axs is iterable when num_images=1

    for idx in range(min(num_images, len(images))):
        img = (
            images[idx].permute(1, 2, 0).clone().detach().cpu().numpy()
        )  # Convert image from (C, H, W) to (H, W, C)
        axs[idx].imshow(img, cmap="gray")  # Display the background image

        if "masks" in targets[idx]:
            all_masks = targets[idx]["masks"]
            for mask in all_masks:
                mask = (
                    mask.squeeze()
                )  # Assuming masks are (1, H, W) and removing singular dimensions
                print(mask.shape)
                rgba_mask = np.zeros((*mask.shape, 4))  # Create an RGBA mask uint8 array

                rgba_mask[:, :, 2] = 1.0  # Blue channel
                mask_data = mask.clone().detach().cpu().numpy()
                rgba_mask[:, :, 3] = (
                    mask_data * 0.1
                )  # Alpha channel, scale mask by 0.5 for transparency

                # Overlay the color mask with transparency where mask values are zero
                axs[idx].imshow(rgba_mask)

        # Draw bounding boxes
        for box in targets[idx]["boxes"]:
            box = box.cpu().numpy()
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            axs[idx].add_patch(rect)

        axs[idx].axis("off")
    plt.tight_layout()
    return fig

fig = plot_images_with_boxes_and_masks(imgs, targets, num_images=9)
fig.savefig("test.png")
#torchvision.transforms.RandomResizedCrop((480, 640), scale=(0.9, 2.1))(to_pil_image(obs[0][0])).save("test2.png")