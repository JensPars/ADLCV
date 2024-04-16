from setupHF_cache import *
import torch
from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
import numpy as np
from autosam import AutoSamPipeline, compress_and_save
from pathlib import Path
from PIL import Image

root = "data/experiments_cg"
root = Path(root)

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16") #"runwayml/stable-diffusion-v1-5"
pipe.enable_xformers_memory_efficient_attention()
#pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
#classes = ['cat','dog','horse','sheep','cow','elephant', 'bear', 'zebra', 'giraffe']
classes = ['cat']
seeds = np.random.randint(0, 1e12, int(1e3))
paths2img = []
cgs = [7.5]

for cg in cgs:
    for cls in classes:
        # create dir
        path2img = root / str(cg) / cls
        path2img.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            torch.manual_seed(seed)
            path2img = root / str(cg) / cls / f'img_{seed}.jpg'
            paths2img.append(path2img)
            image = pipe(f"A full body shot of a {cls}", guidance_scale=cg, num_inference_steps=40).images[0] #, guidance_scale=0.0, num_inference_steps=1
            image.save(path2img, 'JPEG', quality=90)
        
    

del pipe
# Create an instance of AutoSamPipeline
pipeline = AutoSamPipeline()
# Load the image
for path in paths2img:
    out_path = str(path).replace("img", "mask").replace("jpg", "pkl")
    image = Image.open(path)
    # Process the image using the pipeline
    out = pipeline.forward(image)
    if out is not None:
      compress_and_save(out["mask"], out_path)
    else: # delete img
        path.unlink()
