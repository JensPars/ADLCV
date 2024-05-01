from setupHF_cache import *
import torch
from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image
import numpy as np
from autosam import AutoSamPipeline, compress_and_save
from pathlib import Path
from PIL import Image
from glob import glob
root = "data/experiments_v1-5"
root = Path(root)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16") #"runwayml/stable-diffusion-v1-5"
#pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
seeds = np.random.randint(0, 1e12, int(5e2))
paths2img = []
# create dir

for seed in seeds:
    torch.manual_seed(seed)
    path2img = root / f'img_{seed}.jpg'
    paths2img.append(path2img)
    image = pipe("A full body shot of a cat", guidance_scale=7.5).images[0] #, guidance_scale=0.0, num_inference_steps=1
    image.save(path2img, 'JPEG', quality=90)

del pipe
# Create an instance of AutoSamPipeline
pipeline = AutoSamPipeline()
# Load the image
#paths2imgs = glob('data/experiments_turbo/*.jpg')
for path in paths2img:
    out_path = str(path).replace("img", "mask").replace("jpg", "pkl")
    image = Image.open(path)
    # Process the image using the pipeline
    out = pipeline.forward(image)
    if out is not None:
      compress_and_save(out["mask"], out_path)
    else: # delete img
        path.unlink()
        
    

            
