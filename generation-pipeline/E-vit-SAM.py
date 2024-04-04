import zlib
import pickle
import csv
import zlib
import time
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# segment anything
from efficientvit.sam_model_zoo import create_sam_model
from utils import draw_scatter, draw_binary_mask, cat_images
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from glob import glob


point_coords = [[512 // 2, 512 // 2]]
point_labels = [1]
multimask_output = False
disp_cutout = True
weight_url = "assets/checkpoints/l2.pt"
name = "l2"
paths = glob("data/puppies/*")
class_name = "puppy_viz"
anno_dir = f"{class_name}_anno"
efficientvit_sam = create_sam_model(
    name=name,
    weight_url=weight_url,
)
efficientvit_sam = efficientvit_sam.cuda().eval()
efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
for path in paths:
    raw_image = np.array(Image.open(path).convert("RGB"))
    start_time = time.time()

    efficientvit_sam_predictor.set_image(raw_image)
    masks, p, _ = efficientvit_sam_predictor.predict(
        point_coords=np.array(point_coords),
        point_labels=np.array(point_labels),
        multimask_output=multimask_output,
    )
    print(p)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    if disp_cutout:
        plots = [
            draw_scatter(
                draw_binary_mask(raw_image, binary_mask, (0, 0, 255)),
                point_coords,
                color=["g" if l == 1 else "r" for l in point_labels],
                s=10,
                ew=0.25,
            )
            for binary_mask in masks
        ]
        plots = cat_images(plots, axis=1)
        plt.imshow(plots)
        plt.axis("off")
        plt.title(f"Confidence: {p}")
        # save to anno_dir
        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)

        path2 = os.path.join(anno_dir, path.split("/")[-1])
        plt.savefig(path2, bbox_inches="tight", pad_inches=0)

    else:
        compressed_mask = zlib.compress(masks[0].tobytes(), level=9)
        # Write compressed mask to pickle file
        path2anno = anno_dir + path.split(".")[0] + ".pickle"
        with open(path2anno, "wb") as f:
            pickle.dump(compressed_mask, f)
        img_path = path
        anno_path = path2anno

        with open("data/anno.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([img_path, class_name, anno_path])
