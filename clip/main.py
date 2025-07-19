import torch

from model import CLIPModel
from tokenizer import CLIPTextTokenizer
from loss import InfoNCECriterion
from train import Trainer
from config import CLIPConfig


if __name__ ==  "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize components
    config = CLIPConfig()
    tokenizer = CLIPTextTokenizer(config)
    model = CLIPModel(config, tokenizer)
    criterion = InfoNCECriterion()

    trainer = Trainer(config, model, criterion, device, tokenizer)
    trainer.run()
    