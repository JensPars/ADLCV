# https://github.com/sayakpaul/cmmd-pytorch/blob/main/main.py

# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-efficient MMD implementation in JAX."""
from setupHF_cache import *
import torch

# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


def mmd(x, y):
    """Memory-efficient MMD implementation in JAX.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that the first invocation of this function will be considerably slow due
    to JAX JIT compilation.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    #x = torch.from_numpy(x)
    #y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)



# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding models used in the CMMD calculation."""

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
import numpy as np

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
_CUDA_AVAILABLE = torch.cuda.is_available()


def _resize_bicubic(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images


class ClipEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)

        self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        if _CUDA_AVAILABLE:
            self._model = self._model.cuda()

        self.input_image_size = self.image_processor.crop_size["height"]

    @torch.no_grad()
    def embed(self, images):
        """Computes CLIP embeddings for the given images.

        Args:
          images: An image array of shape (batch_size, height, width, 3). Values are
            in range [0, 1].

        Returns:
          Embedding array of shape (batch_size, embedding_width).
        """

        images = _resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        if _CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        image_embs = self._model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs
    
    

if __name__ == "__main__":
    from setupHF_cache import *
    from eval import MaskedData, SynData
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    
    syndata = SynData(
        img_dir="data/experiments_cg/7.5/cat",
        anno_dir="data/experiments_cg/7.5/cat",
        fid=True,
    )
    print(syndata[0])
    syndata = DataLoader(syndata, batch_size=16)
    
    root = "/work3/s194649/train2017"
    anno = "coco_subset_annotations.json"
    transform = T.Compose([T.Resize([512, 512]),T.ToTensor()])
    realdata = DataLoader(MaskedData(root, anno, transform=transform, categories="dog"), batch_size=16, drop_last=True, collate_fn=lambda x: torch.concatenate(x, dim=0))
    clip = ClipEmbeddingModel()
    real_embeds = []
    for real_batch in realdata:
        real_batch = real_batch.permute(0, 2, 3, 1).numpy()
        real_embed = clip.embed(real_batch)
        real_embeds.append(real_embed)
    syn_embeds = []
    for syn_batch in syndata:
        syn_batch = syn_batch.permute(0, 2, 3, 1).numpy()
        syn_embed = clip.embed(syn_batch)
        syn_embeds.append(syn_embed)
        
    real_embeds = torch.concat(real_embeds, dim=0)
    syn_embeds = torch.concat(syn_embeds, dim=0)
    #print(mmd(real_embeds,syn_embeds))
    mmds = []
    for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 998]:
        print(f'with n = {i}, mmd = ', mmd(real_embeds[:i], syn_embeds[:i]))
