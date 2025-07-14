import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from dataset import CLIPDataset
from model import CLIP
from loss import InfoNCECriterion

class Trainer:
    def __init__(self, model, criterion, optimizer, device, tokenizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

        # Initialize OpenAI's GPT-2 BPE tokenizer
        self.tokenizer = tokenizer

        self.train_dataloader = torch.utils.data.DataLoader(
            CLIPDataset(image_dir='../Images', captions_filepath='../captions.txt', transform=None, tokenizer=self.tokenize_text),
            batch_size=32, shuffle=True, num_workers=4
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            CLIPDataset(image_dir='../Images', captions_filepath='../captions.txt', transform=None, tokenizer=self.tokenize_text),
            batch_size=32, shuffle=False, num_workers=4
        )

    def tokenize_text(self, text):
        token_ids = self.tokenizer.encode(text)[:77]
        # Pad with 0s if shorter than 77
        padded = token_ids + [0] * (77 - len(token_ids))
        return torch.tensor(padded, dtype=torch.long)

    def train_step(self, images, token_ids):
        self.model.train()
        images = images.to(self.device)
        token_ids = token_ids.to(self.device)

        logits = self.model(images, token_ids)
        loss = self.criterion(logits)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def run(self, num_epochs):
        for epoch in range(num_epochs):
            for images, token_ids in self.train_dataloader:
                loss = self.train_step(images, token_ids)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")


if __name__ ==  "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model components
    tokenizer = tiktoken.get_encoding("gpt2")
    model = CLIP(vocab_size=tokenizer.n_vocab, image_dim=768, caption_dim=512, embedding_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = InfoNCECriterion()

    print(f"Model: {model}")
    print(f"Vocabulary size: {tokenizer.n_vocab}")

    trainer = Trainer(model, criterion, optimizer, device, tokenizer)
    trainer.run(num_epochs=2)
    

