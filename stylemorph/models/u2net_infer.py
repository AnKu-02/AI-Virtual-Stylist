# segmentation/u2net_infer.py

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

from segmentation.model.u2net import U2NET  # You will use U2NET model class

def load_model(model_path):
    net = U2NET(3, 1)  # 3 input channels (RGB), 1 output channel (mask)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net

def segment_person(image_path, output_path, model):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        d1, *_ = model(input_tensor)
        mask = d1[0][0].cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = (mask * 255).astype(np.uint8)

    # Resize mask back to original image size
    mask = Image.fromarray(mask).resize(image.size)
    mask.save(output_path)
    print(f"Segmentation saved to {output_path}")
