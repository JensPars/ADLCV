from glob import glob
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
path2imgs = glob("data/experiments_cg/fig/*/*/*.jpg")
path2imgs = sorted(path2imgs)
imgs = [T.ToTensor()(Image.open(img)) for img in path2imgs]
grid_img = torchvision.utils.make_grid(imgs, nrow=9)
grid_img = ToPILImage()(grid_img)
grid_img.save("fig.png")
