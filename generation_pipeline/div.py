if __name__ == "__main__":
    from setupHF_cache import *
import torch
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dino-vits16')
model = AutoModel.from_pretrained('facebook/dino-vits16')




# We have to force return_dict=False for tracing
model.config.return_dict = False




if __name__ == "__main__":
    from setupHF_cache import *
    from eval import MaskedData, SynData
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    root = "data/sdxl-turbo-corrputed/"
    root = "data2/sdxl-turbo/"
    root = "data/data-llm/"
    for cls in ["boat", "car", "bus"]:
        syndata = SynData(
            img_dir=root + cls,
            anno_dir=root + cls,
            fid=True,
        )
        print(len(syndata))
        print(root + cls)
        syndata = DataLoader(syndata, batch_size=16, shuffle=True)
        real_embeds = []
        n_real_imgs = 0    
        syn_embeds = []
        n_syn_imgs = 0
        for syn_batch in syndata:
            inputs = processor(images=syn_batch, return_tensors="pt", do_rescale=False)
            syn_batch = syn_batch.permute(0, 2, 3, 1).numpy()
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_states = outputs[0]
                syn_embeds.append(outputs[1])
                #outputs = model(**inputs)
                #last_hidden_states = outputs[0]
                #syn_embeds.append(traced_outputs[1])
            n_syn_imgs += len(syn_batch)
            if n_syn_imgs > 500:
                break
            print(n_syn_imgs/500.0)
        #traced_outputs
        syn_embeds = torch.concat(syn_embeds, dim=0)
        print(cls)
        print(syn_embeds.std(0).mean())