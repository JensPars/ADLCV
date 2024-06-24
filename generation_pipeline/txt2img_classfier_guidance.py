from setupHF_cache import *
import torch
from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
import numpy as np
from autosam import AutoSamPipeline, compress_and_save
from pathlib import Path
from PIL import Image
# import coco dataset
from pycocotools.coco import COCO
import os
from dotenv import load_dotenv
import yaml
import glob


load_dotenv()
root = "/work3/s194649/coco_turbo"
root = Path(root)


annFile = os.environ.get("COCO_VAL_ANN")
coco = COCO(annFile)
classes_already = glob.glob('/work3/s194649/coco_turbo/7.5/*')
classes_already = [os.path.basename(cls) for cls in classes_already]
category_names = coco.cats
category_names = [category_names[cat]['name'] for cat in category_names.keys()]
category_names = [cat for cat in category_names if cat not in classes_already]

#pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16") #"runwayml/stable-diffusion-v1-5"
#pipe.enable_xformers_memory_efficient_attention()
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
print("X"*10)
#classes = ['cat','dog','horse','sheep','cow','elephant', 'bear', 'zebra', 'giraffe']
classes = category_names
print(classes)
seeds = np.random.randint(0, 1e12, int(1e2))
paths2img = []
cgs = [7.5]

for cg in cgs:
    print(cg)
    for cls in classes:
        print(cls)
        # create dir
        path2img = root / str(cg) / cls
        path2img.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            torch.manual_seed(seed)
            path2img = root / str(cg) / cls / f'img_{seed}.jpg'
            paths2img.append(path2img)
            image = pipe(f"An image of a {cls}", guidance_scale=0.0, num_inference_steps=4, strength=0.25).images[0] #, guidance_scale=0.0, num_inference_steps=1
            image.save(path2img, 'JPEG', quality=90)
        
print("Done")

#del pipe
# Create an instance of AutoSamPipeline
#pipeline = AutoSamPipeline()
# Load the image
#for path in paths2img:
#    out_path = str(path).replace("img", "mask").replace("jpg", "pkl")
#    image = Image.open(path)
#    # Process the image using the pipeline
#    out = pipeline.forward(image)
#    if out is not None:
#      compress_and_save(out["mask"], out_path)
#    else: # delete img
#        path.unlink()
