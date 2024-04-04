import os
import pickle
import zlib
os.environ['HF_HOME'] = '/work3/s194649/cache'
os.environ['TRANSFORMERS_CACHE'] = "/work3/s194649/cache"
from glob import glob
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
import numpy as np


class AutoSamPipeline:
    def __init__(self,):
        name = "l2"
        weight_url = "assets/checkpoints/l2.pt"
        self.detector = pipeline(model="facebook/detr-resnet-50", device="cuda")
        efficientvit_sam = create_sam_model(
        name=name,
        weight_url=weight_url,
    )
        efficientvit_sam = efficientvit_sam.cuda().eval()
        self.efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
     
    def forward(self, img):
        out = self.detector(img)
        box = out[np.argmax([o['score'] for o in out if o['label']=='dog'])]['box']
        # Convert box to the format expected by EfficientViTSamPredictor
        # Typically, it's [xmin, ymin, xmax, ymax]
        box = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
        # Convert to numpy array
        box = np.array(box)
        # Use EfficientViTSamPredictor to predict mask
        raw_image = np.array(img.convert("RGB"))
        self.efficientvit_sam_predictor.set_image(raw_image)
        masks, iou_predictions, _ = self.efficientvit_sam_predictor.predict(box=box, multimask_output=False)
        return {"mask": masks[0], "iou_pred": iou_predictions, "bbox": box}

def visualize_and_save(box, mask, img, out_file):
    # Load image
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.imshow(mask, alpha=(mask.astype(float) * 0.5))
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # Save to memory
    plt.savefig(out_file, format='png')

def compress_and_save(mask, outfile):
    compressed_mask = zlib.compress(mask.tobytes(), level=9)
    # Write compressed mask to pickle file
    with open(outfile, "wb") as f:
        pickle.dump(compressed_mask, f)

def read_and_decompress(file):
    # Read the compressed mask from the pickle file
    with open(file, "rb") as f:
        compressed_mask = pickle.load(f)
    # Decompress the mask
    decompressed_mask = zlib.decompress(compressed_mask)
    # Convert the decompressed mask to numpy array
    mask = np.frombuffer(decompressed_mask, dtype=np.bool_)
    n = np.sqrt(len(mask)).astype(int)
    mask = mask.reshape((n, n))
    return mask

if __name__ == "__main__":
    # Create an instance of AutoSamPipeline
    pipeline = AutoSamPipeline()
    # Load the image
    image_path = glob("data/puppies/*")[0]
    image = Image.open(image_path)
    # Process the image using the pipeline
    out = pipeline.forward(image)
    # Visualize and save the output
    visualize_and_save(out["bbox"], out["mask"], image, "output.png")
    # Compress the mask
    compress_and_save(out["mask"], "compressed_mask.pkl")
    # Read and decompress the mask
    decompressed_mask = read_and_decompress("compressed_mask.pkl")
    # Plot the decompressed mask
    visualize_and_save(out["bbox"], decompressed_mask, image, "output_alt.png")
    