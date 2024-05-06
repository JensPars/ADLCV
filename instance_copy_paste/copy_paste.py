import torch
import numpy as np
import random
from PIL import Image
from typing import List, Optional
import albumentations as A
from skimage.filters import gaussian

class InstanceRetriever():
    def __init__(self, instance_pool:torch.utils.data.Dataset):
        self.instance_pool = instance_pool

    def get_instances(self, n_instances:int):
        n_instances = random.randint(1, n_instances) 
        #? Sample the indicies such without replacement, such that it is unique instances. 
        instances = []
        for i in range(n_instances):
            instances.append(self.instance_pool[random.randint(0, len(self.instance_pool)-1)])
        return instances
class InstanceCopyPaste():

    def __init__(self, instance_retriever:InstanceRetriever, layering:str, n_instances:int, min_visible_pct:float=1.0, min_visible_keyspoint:int = 10):
        self.instance_retriever = instance_retriever
        self.min_visible_pct = min_visible_pct
        self.n_instances = n_instances
        self.layering = layering # TODO: Implement the layering
        self.min_visible_keyspoint = min_visible_keyspoint

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
            background = self._blend_masks2background(image, background, adjusted_masks, sigma = 10.0)

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
        
        # Compute visibility of keypoints per mask
        visible_keypoints_per_mask = final_masks.sum(axis=(1, 2))

        # Adjust the bounding boxes and remove masks with too few key points visible
        adjusted_bboxes = []
        adjusted_masks = []
        adjusted_labels = []
        #assert len(new_masks) == len(bboxes), "Number of masks and bounding boxes do not match."
        assert len(new_masks) == len(visible_keypoints_per_mask), "Number of masks and visible keypoints do not match."
        
        for i in range(len(visible_keypoints_per_mask)):
            if visible_keypoints_per_mask[i] > self.min_visible_keyspoint:
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
    