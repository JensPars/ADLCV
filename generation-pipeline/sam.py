from transformers import pipeline
generator =  pipeline("mask-generation", device="cuda", points_per_batch = 256)
image_url = "/zhome/ca/9/146686/ADLCV/puppies/output2.jpg"
outputs = generator(image_url, points_per_batch = 1)