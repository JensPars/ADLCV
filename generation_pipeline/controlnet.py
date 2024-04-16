from setupHF_cache import *
import numpy as np
import matplotlib.pyplot as plt
from generation_pipeline.coco.coco_masks import get_instance_masks

animal =  np.array([[255, 0, 122]])
classes = ['cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']


from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
        )
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
seeds = np.load('generation_pipeline/random_seeds.npy')[:10]
for dyr in classes:
    iterator = get_instance_masks(coco_annotation_file= "coco_subset_annotations.json", class_name=dyr)
    for seed, instance_mask in zip(seeds, iterator):
        color_seg = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8) # height, width, 3
        color_seg = color_seg.astype(np.uint8)
        color_seg[instance_mask==1] = animal
        image = Image.fromarray(color_seg)
        image = pipe(dyr, image, num_inference_steps=20).images[0]
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image)
        axs[0].imshow(instance_mask, alpha=(instance_mask.astype(float)*0.4), cmap="bwr")
        axs[0].set_title('mask')
        axs[1].imshow(image)
        axs[1].set_title('Generated Image')
        axs[0].axis('off')
        axs[1].axis('off')
        plt.tight_layout()
        plt.savefig(f'data/AB_ablation/{dyr}{seed}.jpg')

