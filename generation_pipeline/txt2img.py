from setupHF_cache import *
import torch
from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image
import numpy as np
from autosam import AutoSamPipeline, compress_and_save
from pathlib import Path
from PIL import Image

root = "data/experiments_cg"
root = Path(root)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16) #"runwayml/stable-diffusion-v1-5"
#pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
classes = ['cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']
seeds = np.load('generation_pipeline/random_seeds.npy')[:10]
paths2img = []

for cg in [1, 7.5, 15]:
    for cls in classes:
        # create dir
        path2img = root / str(cg) / cls
        path2img.mkdir(parents=True, exist_ok=True)
        
        for seed in seeds:
            torch.manual_seed(seed)
            path2img = root / str(cg) / cls / f'img_{seed}.jpg'
            paths2img.append(path2img)
            image = pipe(cls, guidance_scale=cg).images[0] #, guidance_scale=0.0, num_inference_steps=1
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
        
    

            
