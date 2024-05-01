from setupHF_cache import *
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
import numpy as np
from autosam import AutoSamPipeline, compress_and_save
from pathlib import Path
from PIL import Image
from glob import glob
model_type = "stabilityai/sdxl-turbo"
#pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")  #StableDiffusionPipeline.from_pretrained(model_type, torch_dtype=torch.float16, variant="fp16") #"runwayml/stable-diffusion-v1-5"
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
seeds = np.random.randint(0, 1e12, int(5e3))
paths2img = []
root = Path("data") / model_type.split("/")[-1]
# create dir
cats = ["bus"] #["car"] #"boat", 
wn_cats = {"car":['car', 'taxi', 'taxicab', 'beach_wagon', 'station_wagon', 'compact car', 'convertible car', 'coupe car', 'police cruiser', 'police car', 'hatch back', 'sports car', 'supercar', 'used car', 'SUV', 'minivan'],
 "bus": ['bus', 'autobus', 'coach bus', 'double-decker bus', 'jitney bus', 'motorbus', 'motorcoach'],
 "boat": ['boat', 'ship', 'catamaran', 'cargo ship', 'sailboat', 'icebreaker ship', 'destroyer ship', 'gondola', 'motorboat', 'tender boat', 'lugger boat', 'ferry', 'pirate ship', 'row boat', 'barge boat', 'cruise ship', 'yacht']}

for cat in cats:
  cat_root = root / cat
  # sample random from 
  if not cat_root.exists():
      cat_root.mkdir()
  for seed in seeds:
    name = np.random.choice(wn_cats[cat])
    torch.manual_seed(seed)
    path2img = cat_root / f'img_{name}_{seed}.jpg'
    paths2img.append(path2img)
    image = pipe(f"An image of a {name}", guidance_scale=0., num_inference_steps=4, strength=0.25).images[0] #, guidance_scale=0.0, num_inference_steps=1 photorealistic
    image.save(path2img, 'JPEG', quality=90)
    break
  break

del pipe
# Create an instance of AutoSamPipeline
pipeline = AutoSamPipeline()
# Load the image
paths2img = glob('data/sdxl-turbov2_4steps/car/*.jpg')
for path in paths2img:
    path = Path(path)
    out_path = str(path).replace("img", "mask").replace("jpg", "pkl")
    image = Image.open(path)
    # Process the image using the pipeline
    out = pipeline.forward(image)
    if out is not None:
      compress_and_save(out["mask"], out_path)
    else: # delete img
        path.unlink()
        
    

            
