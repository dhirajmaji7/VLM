import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use TkAgg backend for plotting

class CLIPDataset(Dataset):
    def __init__(self, image_dir, captions_filepath, tokenizer=None):
        super().__init__()
        self.image_dir = image_dir
        self.captions_filepath = captions_filepath
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.captions_dict = {}
        self.images = {}
        self.read_captions(captions_filepath)   
        self.read_images()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_fname = list(self.images.keys())[idx]
        img = self.images[image_fname]
        caption = self.captions_dict.get(image_fname, [])[0]  # TODO: Handle multiple captions
        if self.tokenizer:
            caption = self.tokenizer(caption)
        return img, caption

    def read_captions(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname_caption = line.split(',')
                fname = fname_caption[0].split()[0]  # e.g. 1000268201_693b08cb0e.jpg
                if not fname.lower().endswith('.jpg'):
                    continue                
                caption = fname_caption[1]
                self.captions_dict.setdefault(fname, []).append(caption)
    
    def read_images(self):
        for fname in os.listdir(self.image_dir):
            if fname.lower().endswith('.jpg'):
                image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
                image = self.transform(image)
                self.images[fname] = image


# clipdataset = CLIPDataset(image_dir='../Images',captions_filepath='../captions.txt', transform=None)
# img, caption = clipdataset.__getitem__(7)
# print(caption)
# plt.imshow(img)  # Convert from CxHxW to HxWxC for plotting
# plt.axis('off')  # Hide axes
# plt.show()


