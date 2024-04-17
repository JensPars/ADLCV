import torch
import numpy as np
import random
from PIL import Image
from typing import List, Optional
import albumentations as A
class InstanceRetriever():
    def __init__(self, instance_pool:torch.utils.data.Dataset):
        self.instance_pool = instance_pool

    def get_instances(self, n_instances:int):
        # n_instances = random.randint(1, n_instances) #? Maybe to vary the number of instances pasted
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
    def __call__(self, image, masks, bboxes):
        return self.copy_paste(image, masks, bboxes)

    def copy_paste(self, image, masks, bboxes):
        """

        """
        # Background Image
        background = Image.fromarray(image)
        # Collect new instances to paste onto the background
        instances = self.instance_retriever.get_instances(self.n_instances)
        new_masks = []
        new_bboxes = []

        for instance in instances:
            background, new_instance_mask, new_instance_bbox = self._paste_single_instance(instance['image'], instance['masks'][0], background)
            new_instance_bbox = new_instance_bbox + [instance['bboxes'][0][4]] # Add label to the bounding box
            new_masks.append(new_instance_mask)
            new_bboxes.append(new_instance_bbox)
        #print(len(new_masks), len(new_bboxes))
        #print(bboxes)
        bboxes.extend(new_bboxes)
        masks.extend(new_masks)
        #print(len(masks), len(bboxes))
        adjusted_masks, adjusted_bboxes = self._adjust_masks_and_bboxes(masks, bboxes)
        
        return {"image": np.array(background), "masks": adjusted_masks, "bboxes": adjusted_bboxes}

    def _adjust_masks_and_bboxes(self, masks, bboxes):
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
        assert len(new_masks) == len(bboxes), "Number of masks and bounding boxes do not match."
        assert len(new_masks) == len(visible_keypoints_per_mask), "Number of masks and visible keypoints do not match."
        
        for i in range(len(visible_keypoints_per_mask)):
            if visible_keypoints_per_mask[i] > self.min_visible_keyspoint:
                adjusted_masks.append(new_masks[i])
                x, y, width, height = self.extract_bbox(new_masks[i])
                adjusted_bboxes.append([(x, y, width, height, bboxes[i][4])])
        
        return adjusted_masks, adjusted_bboxes
    
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
        return int(xmin), int(ymin), int(xmax)-int(xmin), int(ymax)-int(ymin)
      
    # TODO: Implement the layering
    #! DOES NOT WORK YET
    def _collect_bg_instance(self, image, masks, bboxes):
        """
        Collect background instances such these 
        can be pasted onto the background 
        image on top of other new instances.
        """
        insts = zip(masks, bboxes)
        bg_insts = []
        for mask, bbox in insts:
            cropped_image = self._crop_image(image, bbox)
            cropped_mask = self._crop_image(mask, bbox)

            bg_insts.append({
                'image': cropped_image,
                'masks': cropped_mask,
                'bboxes': bbox
            })
        return bg_insts

    def _crop_image(self, image, bbox):
        """
        Crop an image given a bounding box.
        """
        x, y, width, height = bbox
        return image[y:y+height, x:x+width]

    # TODO: There is a bug in the pasting of the image
    # TODO: sometimes it will have a mismatch with the effective mask and so fort
    def _paste_single_instance(self, image, mask, background):
        """
        Paste a segmented part of an image such that at least 25% of it remains within
        the background image boundaries.
        
        Args:
        image (PIL.Image): Input image of size n by n.
        mask (numpy.array): Segmentation mask, same dimensions as image.
        bbox (tuple): Bounding box (xmin, ymin, xmax, ymax).
        background (PIL.Image): Background image to paste onto.
        
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
        segmented_part = Image.composite(image, Image.new('RGB', image.size), mask_image)

        # Calculate the range for the random location
        # Allowing part of the image to be outside the background
        min_x = int(-(1-self.min_visible_pct) * image.width)
        max_x = int(background.width - self.min_visible_pct * image.width)
        min_y = int(-(1-self.min_visible_pct) * image.height)
        max_y = int(background.height - self.min_visible_pct * image.height)

        random_x = random.randint(min_x, max_x)
        random_y = random.randint(min_y, max_y)

        # Paste the segmented part onto the background
        background.paste(segmented_part, (random_x, random_y), mask_image)

        # Update the mask and bounding box
        new_mask = np.zeros((background.height, background.width), dtype=np.uint8)
        #! This is where the bug is (below)
        # Calculate the effective coordinates considering possible negative indices
        effective_x = max(random_x, 0)
        effective_y = max(random_y, 0)
        effective_width = min(segmented_part.width, background.width - effective_x)
        effective_height = min(segmented_part.height, background.height - effective_y)
        #print("New Mask Indices:", effective_y, effective_y + effective_height, effective_x, effective_x + effective_width)
        #print("Mask Indices:",  max(-random_y, 0),  max(-random_y, 0) + effective_height, max(-random_x, 0), max(-random_x, 0) + effective_width)

        new_mask[effective_y:effective_y + effective_height, effective_x:effective_x + effective_width] = mask[
            max(-random_y, 0):max(-random_y, 0) + effective_height,
            max(-random_x, 0):max(-random_x, 0) + effective_width
        ]

        new_bbox = [
            effective_x,
            effective_y,
            effective_x + effective_width,
            effective_y + effective_height
        ]
        return background, new_mask, new_bbox