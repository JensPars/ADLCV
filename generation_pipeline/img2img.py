import torch
from diffusers import AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline
from eval import MaskedData, SynData
import torchvision.transforms as T
# import tensor to PIL
from torchvision.transforms import ToPILImage, PILToTensor
# import torchvision image grid
from torchvision.utils import make_grid

root = "/work3/s194649/train2017"
anno = "./subsets/instances_train2017_subset_0.1.json"
transform = T.Compose([T.Resize([512, 512]), T.ToTensor()])
realdata = MaskedData(root, anno, transform=transform, categories="dog")
img = realdata[0]

ToImg = ToPILImage()
ToTens = PILToTensor()
img = ToImg(img)
img.save("dog.png")
imgs = [realdata[i] for i in range(10)]
# make grid
grid = make_grid(imgs, nrow=5)
ToImg(grid).save("dogs_grid.jpg")
#ToImg(imgs[0]).save("dog1.png")
# if avail
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

synth_imgs = []
prompt = "An image of a dog"
for img in imgs:
    images = pipe(prompt=prompt, image=ToImg(img), strength=0.75, guidance_scale=2).images
    synth_imgs.append(ToTens(images[0]))

grid = make_grid(synth_imgs, nrow=5)
ToImg(grid).save("dogs_synth0.75.jpg")



