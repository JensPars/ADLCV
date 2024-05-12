from setupHF_cache import *
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
import numpy as np
from autosam import AutoSamPipeline, compress_and_save
from pathlib import Path
from PIL import Image
from glob import glob
import json
model_type = "stabilityai/sdxl-turbo"
#pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")  #StableDiffusionPipeline.from_pretrained(model_type, torch_dtype=torch.float16, variant="fp16") #"runwayml/stable-diffusion-v1-5"
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
seeds = np.random.randint(0, 1e12, int(5e3))
paths2img = []
root = Path("data-llm") #/ model_type.split("/")[-1]
# create dir
cats = ["bus", "car", "boat"] 
wn_cats = {"car":['car', 'taxi', 'taxicab', 'beach_wagon', 'station_wagon', 'compact car', 'convertible car', 'coupe car', 'police cruiser', 'police car', 'hatch back', 'sports car', 'supercar', 'used car', 'SUV', 'minivan'],
 "bus": ['bus', 'autobus', 'coach bus', 'double-decker bus', 'jitney bus', 'motorbus', 'motorcoach'],
 "boat": ['boat', 'ship', 'catamaran', 'cargo ship', 'sailboat', 'icebreaker ship', 'destroyer ship', 'gondola', 'motorboat', 'tender boat', 'lugger boat', 'ferry', 'pirate ship', 'row boat', 'barge boat', 'cruise ship', 'yacht']}

llm_cats = json.load(open("noun_descriptions1.json"))
ids = {"car": 3, "bus": 6, "boat": 9}

for cat in cats:
    cat_root = root / cat
    if not cat_root.exists():
        cat_root.mkdir()
    i=0
    for v in llm_cats[cat]:
        #v = [item for sublist in v for item in sublist]
        for d in v:
           print(d)
           path2img = cat_root / f'img_{i}.jpg'
           print(path2img)
           paths2img.append(path2img)
           image = pipe(d, guidance_scale=0., num_inference_steps=4, strength=0.25).images[0]
           image.save(path2img, 'JPEG', quality=90)
           i+=1
           
    # Create an instance of AutoSamPipeline
    pipeline = AutoSamPipeline()
    # Load the image
    paths2img = glob(f'data-llm/{cat}/*.jpg')
    for path in paths2img:
        path = Path(path)
        out_path = str(path).replace("img", "mask").replace("jpg", "pkl")
        image = Image.open(path)
        # Process the image using the pipeline
        out = pipeline.forward(image, cat)
        if out is not None:
          compress_and_save(out["mask"], out_path)
        else: # delete img
            path.unlink()
        
    

            
