import torch

from model import CLIPModel
from tokenizer import CLIPTextTokenizer
from loss import InfoNCECriterion
from train import Trainer
from config import CLIPConfig


if __name__ ==  "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize components
    config = CLIPConfig()
    tokenizer = CLIPTextTokenizer(config)
    model = CLIPModel(config)
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config.learning_rate, 
                                weight_decay=config.weight_decay)
    criterion = InfoNCECriterion()

    trainer = Trainer(config, model, criterion, optimizer, device, tokenizer)
    print("Starting training...")
    trainer.run()
    