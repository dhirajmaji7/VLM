import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CLIPDataset
from model import CLIP
from tokenizer import CLIPTextTokenizer
from loss import InfoNCECriterion
from utils import timeit

class Trainer:
    def __init__(self, model, criterion, optimizer, device, tokenizer, num_epochs):
        self.device = device
        self.model = model.to(self.device)
        #self.criterion = criterion.to(self.device)
        self.criterion = F.cross_entropy
        self.optimizer = optimizer
        self.num_epochs = num_epochs

        # Initialize OpenAI's GPT-2 BPE tokenizer
        self.tokenizer = tokenizer

        self.train_dataloader = torch.utils.data.DataLoader(
            CLIPDataset(image_dir='../Images', captions_filepath='../captions.txt', tokenizer=self.tokenizer.tokenize_text),
            batch_size=32, shuffle=True, num_workers=4
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            CLIPDataset(image_dir='../Images', captions_filepath='../captions.txt', tokenizer=self.tokenizer.tokenize_text),
            batch_size=32, shuffle=False, num_workers=4
        )

    def train_step(self, images, token_ids):
        images = images.to(self.device)
        token_ids = token_ids.to(self.device)
        #print("token id",token_ids)
        logits = self.model(images, token_ids)
        target = torch.arange(logits.size(0), device=self.device)  # B
        loss = self.criterion(logits, target)

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


if __name__ ==  "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model components
    tokenizer = CLIPTextTokenizer(context_length=25)
    model = CLIP(vocab_size=tokenizer.n_vocab, image_dim=192, caption_dim=512, embedding_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = InfoNCECriterion()

    trainer = Trainer(model, criterion, optimizer, device, tokenizer, num_epochs=10)
    print("Starting training...")
    trainer.run()
    

