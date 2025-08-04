import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime

from dataset import CLIPDataset
from config import CLIPConfig
from utils import timeit

class Trainer:
    def __init__(self, config, model, train_dataloader, val_dataloader, criterion, device, tokenizer):
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                lr=config.learning_rate, 
                                weight_decay=config.weight_decay)
        self.tokenizer = tokenizer
        self.num_epochs = config.num_epochs

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        os.makedirs(config.runs_dir, exist_ok=True)
        self.run_dir = os.path.join(config.runs_dir, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"Current training run directory created at: {self.run_dir}")
        self.checkpoint_dir = os.path.join(self.run_dir, config.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory created at: {self.checkpoint_dir}")

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss, self.best_epoch = float('inf'), 0
    
    @timeit
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        iter = 0
        
        for images, token_ids in self.train_dataloader:
            images = images.to(self.device)
            token_ids = token_ids.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images, token_ids)
            loss = self.criterion(logits)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            print(f"Training Batch [{iter + 1}/{len(self.train_dataloader)}]: Loss {loss.item()}", end='\r')
            iter += 1
        return total_loss / len(self.train_dataloader)
    
    @timeit
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        iter = 0
        with torch.no_grad():
            for images, token_ids in self.val_dataloader:
                images = images.to(self.device)
                token_ids = token_ids.to(self.device)

                logits = self.model(images, token_ids)
                loss = self.criterion(logits)
                total_loss += loss.item()
                print(f"Validation Batch [{iter + 1}/{len(self.val_dataloader)}]: Loss {loss.item()}", end='\r')
                iter += 1
        return total_loss / len(self.val_dataloader)

    @timeit
    def run(self):
        print("Starting training...")
        print(f"Using device: {self.device}")

        for epoch in range(self.num_epochs):
            print(f"*********  Epoch {epoch + 1}/{self.num_epochs}  *********")
            
            self.train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1} - Train Loss: {self.train_loss:.4f}")
            self.val_loss = 0 #self.validate()
            print(f"Epoch {epoch + 1} - Validation Loss: {self.val_loss:.4f}")
            self.train_losses.append(self.train_loss)
            self.val_losses.append(self.val_loss)

            # Save best model based on validation loss
            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.run_dir + "/best_model.pth")
                print(f"Best model saved at epoch {self.best_epoch} with validation loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint every `checkpoint_interval` epochs
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                torch.save(self.model.state_dict(), self.checkpoint_dir + f"/checkpoint_epoch_{epoch+1}.pth")
                print(f"Model checkpoint saved at epoch {epoch + 1} to {self.checkpoint_dir}")
            
        # Save final model
        print("********* Training complete ********")
        torch.save(self.model.state_dict(), self.run_dir + "/final_model.pth")
        print(f"Final model saved to {self.run_dir}/final_model.pth")
        
        self.save_results()
        self.print_summary()
    

    def save_results(self):
        results = {
            "config": self.config.__dict__,
            "run_dir": self.run_dir,
            "total_epochs": self.config.num_epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_model": f"{self.run_dir}/best_model.pth",
            "final_model": f"{self.run_dir}/final_model.pth"
        }
        results_path = os.path.join(self.run_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Training results saved to: {results_path}")
    
    def print_summary(self):
        print("********* Training Summary *********")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Final training loss: {self.train_loss:.4f}")
        print(f"Final validation loss: {self.val_loss:.4f}")
        print(f"Final model: {self.run_dir}/final_model.pth")
        print("-" * 10)
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model: {self.run_dir}/best_model.pth")


