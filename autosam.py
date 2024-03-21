import os
os.environ['TRANSFORMERS_CACHE'] = "/work3/s194649/cache"
os.environ['HF_HOME'] = '/work3/s194649/cache'
from glob import glob
from transformers import pipeline
from PIL import Image

def main():
    detector = pipeline(model="facebook/detr-resnet-50", device="cuda")
    path2imgs = "puppies/*"
    for img in glob(path2imgs):
        image = Image.open(img)
        # Process the image here
        out = detector(img)
        break
    visualize_and_save(out,image, "pred.png")



import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

def visualize_and_save(output, img, out_file):
    # Load image
    #img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw bounding boxes
    for result in output:
        x, y, xmax, ymax = result['box']['xmin'], result['box']['ymin'], result['box']['xmax'], result['box']['ymax']
        width = xmax - x
        height = ymax - y
        label = result['label']
        score = result['score']
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f'{label} {score:.2f}', color='white', bbox=dict(facecolor='red', edgecolor='red'))

    # Save to memory
    plt.savefig(out_file, format='png')



if __name__ == "__main__":
    main()