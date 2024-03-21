import os
os.environ['TRANSFORMERS_CACHE'] = "/work3/s194649/cache"
os.environ['HF_HOME'] = '/work3/s194649/cache'
import torch
from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image



#pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16) #"runwayml/stable-diffusion-v1-5"
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
#pipe.unet.to(memory_format=torch.channels_last)
#pipe.vae.to(memory_format=torch.channels_last)
#pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
#pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
for i in range(1000):
    prompt = "a photo of a dog, full-body shot"
    image = pipe(prompt, guidance_scale=0.0, num_inference_steps=1).images[0]
    image.save(f"puppies/output{i}.jpg", 'JPEG', quality=90)