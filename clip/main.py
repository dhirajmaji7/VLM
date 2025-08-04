import torch

from model import CLIPModel
from tokenizer import CLIPTextTokenizer
from loss import InfoNCECriterion
from train import Trainer
from config import CLIPConfig
from dataset import CLIPDataset
from torch.utils.data import DataLoader


if __name__ ==  "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = CLIPConfig()
    tokenizer = CLIPTextTokenizer(config)

    train_dataloader = DataLoader(
        CLIPDataset(config, 'train', tokenizer=tokenizer.tokenize_text),
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    val_dataloader = DataLoader(
            CLIPDataset(config, 'val', tokenizer=tokenizer.tokenize_text),
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers
        )

    # Initialize components
   
    model = CLIPModel(config)
    criterion = InfoNCECriterion()
  

    trainer = Trainer(config, model,train_dataloader,val_dataloader, criterion, device, tokenizer)
    trainer.run()
    