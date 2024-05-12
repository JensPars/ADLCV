from glob import glob
from PIL import Image
from torchvision.utils import make_grid
from numpy import random
# to tensor
from torchvision.transforms import ToTensor
# import tensor to PIL
from torchvision.transforms import ToPILImage

tens = ToTensor()
pil = ToPILImage()
imgpaths = []
for name in ["car", "boat", "bus"]:
    imgdir = f"data/data-llm/{name}/*.jpg"
    imgpath = glob(imgdir)
    # sample random 10 images
    imgpaths += list(random.choice(imgpath, 10))
    
imgs = []
for img in imgpaths:
    img = Image.open(img)
    imgs.append(tens(img))
    
    
grid = make_grid(imgs, nrow=10)
grid = pil(grid)
grid.save("data/boat_grid.jpg")