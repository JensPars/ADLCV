# Project: Instance Segmentation with Augmentation

## 2. Data

In this project, you can use the COCO dataset [6], as standard for the instance segmentation task. Since COCO has 120K images, you can use either a subset of COCO (e.g., minicoco) or a smaller dataset (e.g., PASCAL VOC).

## 3. Tasks

In this project, you could work on the following tasks:

### Task 1: Reproduce a simple Copy-Paste augmentation algorithm [3]
Evaluate the results of this augmentation algorithm on a standard instance segmentation problem. You can also compare the results with several other baselines, such as standard augmentation, RandAugment, or AutoAugment.
https://github.com/open-mmlab/mmdetection/tree/master/configs/simple_copy_paste
https://mmdetection.readthedocs.io/en/v2.25.0/_modules/mmdet/datasets/pipelines/transforms.html

### Task 2: Generate new instances
Use Stable diffusion or ControlNet to generate new instances for the target object categories.

### Task 3: Segment new instances
Use SAM or something else to generate pseudo-GT masks for the generated instances. Alternatively, you can experiment with the new Guided Diffusion model [?].

### Task 4: Copy-Paste and Stable Diffusion
Combine the generated instances with the Copy-Paste strategy and augment your training set with more object instances.

### Brain-storm Task 5: In-painting? Scaling laws? Additional control? LLM generation for captions? etc.

# Reading list
Learning Vision from Models Rivals Learning Vision from Data
https://github.com/google-research/syn-rep-learn/blob/main/SynCLR/assets/synclr_paper.pdf

Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation
https://arxiv.org/pdf/2012.07177.pdf

MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation
https://arxiv.org/pdf/2309.13042.pdf

X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion
https://arxiv.org/pdf/2212.03863.pdf

Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection
https://arxiv.org/pdf/1708.01642.pdf

Cut and Learn for Unsupervised Object Detection and Instance Segmentation
https://arxiv.org/pdf/2301.11320

Beyond Generation: Harnessing Text to Image Models for Object Detection and Segmentation
https://arxiv.org/pdf/2309.05956.pdf
