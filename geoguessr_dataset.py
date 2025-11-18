import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os, os.path
from PIL import Image
import json

class GeoGuessrDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.meta = []
        with open(os.path.join(data_dir, 'meta.jsonl'), 'r') as f:
            for line in f:
                self.meta.append(json.loads(line))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_dir, meta["img"])
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                normalize,
            ])
        
        img = pil_loader(img_path)
        data = transform(img)
        
        target = {"lat": meta["lat"], "lon": meta["lon"]}
        
        return data, target
    
def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")