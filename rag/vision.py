from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_image_embedding(image_file):
    image = Image.open(image_file).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # 🔥 Convert safely to numpy
    if isinstance(outputs, torch.Tensor):
        embedding = outputs.cpu().numpy()
    else:
        embedding = outputs[0].cpu().numpy()

    return embedding[0]
