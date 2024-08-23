import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def vis_image_mask(image, target):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    #plt.figure(facecolor='black')
    img = (
        image.permute(1, 2, 0).clone().detach().cpu().numpy()
    )  # Convert image from (C, H, W) to (H, W, C)
    axs[0].imshow((img * 255).astype(np.uint8))  # Display the background image
    axs[1].imshow((img * 255).astype(np.uint8))
    axs[1].imshow(target["masks"].argmax(0), alpha=target["masks"].any(0).to(float)*0.5, cmap="tab10")  # Display the mask
    axs[0].axis("off")
    axs[1].axis("off")
    axs[0].set_title("Image")
    axs[1].set_title("Mask")
    # Draw bounding boxes
    for box in target["boxes"]:
        box = box.cpu().numpy()
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        #axs[0].add_patch(rect)
        axs[1].add_patch(rect)
    # set tight layout
    plt.tight_layout()
    return fig
    
    

import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))