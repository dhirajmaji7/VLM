import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import os
import random
from PIL import Image
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use TkAgg backend for plotting

from config import CLIPConfig

class CLIPDataset(Dataset):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        random.seed(config.random_seed)
        self.image_dir = config.image_dir
        self.captions_filepath = config.captions_filepath
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.captions_dict = self.read_captions(self.captions_filepath)   
        self.images_dict = self.read_images(self.image_dir)

    def __len__(self):
        return len(self.images_dict)
    
    def __getitem__(self, idx):
        image_fname = list(self.images_dict.keys())[idx]
        img = self.images_dict[image_fname]
        img = self.transform(img)
        captions = self.captions_dict.get(image_fname, [])
        caption = random.choice(captions) if captions else ""
        if self.tokenizer:
            caption = self.tokenizer(caption)
        return img, caption

    def read_captions(self, filename):
        captions_dict = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname_caption = line.split(',')
                fname = fname_caption[0].split()[0]
                if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
                    print(f"Skipping non-image file: {fname}")
                    continue                
                caption = fname_caption[1]
                captions_dict.setdefault(fname, []).append(caption)
        return captions_dict
    
    def read_images(self, image_dir):
        images_dict = {}
        for fname in os.listdir(image_dir):
            if fname.lower().endswith('.jpg') or fname.lower().endswith('.png'):
                image = Image.open(os.path.join(image_dir, fname)).convert('RGB')
                images_dict[fname] = image
        return images_dict


def visualize_sample(clipdataset, idx):
    img, caption = clipdataset.__getitem__(idx)
    print(caption)
    img = img.permute(1, 2, 0).numpy() # Convert to HWC format for plotting
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    config = CLIPConfig()
    clipdataset = CLIPDataset(config=config, tokenizer=None)
    visualize_sample(clipdataset, idx=0)
