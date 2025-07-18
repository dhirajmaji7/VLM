import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CLIPDataset
from config import CLIPConfig
from utils import timeit

class Trainer:
    def __init__(self, config, model, criterion, optimizer, device, tokenizer):
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.num_epochs = config.num_epochs

        self.train_dataloader = DataLoader(
            CLIPDataset(config, tokenizer=self.tokenizer.tokenize_text),
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers
        )
        self.val_dataloader = DataLoader(
            CLIPDataset(config, tokenizer=self.tokenizer.tokenize_text),
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers
        )

    def train_step(self, images, token_ids):
        images = images.to(self.device)
        token_ids = token_ids.to(self.device)
        logits = self.model(images, token_ids)
        loss = self.criterion(logits)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    @timeit
    def train_epoch(self, epoch):
        self.model.train()
        for i, (images, token_ids) in enumerate(self.train_dataloader):
            loss = self.train_step(images, token_ids)
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_dataloader)}], Loss: {loss:.4f}")
        return loss

    @timeit
    def run(self):
        for epoch in range(self.num_epochs):
            loss = self.train_epoch(epoch)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss:.4f}")

