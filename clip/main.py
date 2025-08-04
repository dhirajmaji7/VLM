import torch
import wandb

from model import CLIPModel
from tokenizer import CLIPTextTokenizer
from loss import InfoNCECriterion
from train import Trainer
from config import CLIPConfig


if __name__ ==  "__main__":
    wandb.login()
    wandb_run = wandb.init(entity="dhirajmaji7-student", project="clip", reinit=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize components
    config = CLIPConfig()
    tokenizer = CLIPTextTokenizer(config)
    model = CLIPModel(config, tokenizer)
    criterion = InfoNCECriterion()

    trainer = Trainer(config, model, criterion, device, tokenizer)
    wandb_run.watch(model, criterion, log="all", log_freq=10)
    trainer.run(wandb_run)

    wandb_run.finish()
    