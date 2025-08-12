import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import Blip2Config
from dataset import Blip2Dataset
from tokenizer import FlanT5Tokenizer, BertTokenizer
from model import Blip2Model
from loss import ITCLoss, ITMLoss, ITGLoss

class Blip2Trainer:
    def __init__(self):
        self.config = Blip2Config()
        self.setup_tokenizers()
        self.setup_datasets()
        self.model = Blip2Model(self.config)
        print(self.model)
        self.device = torch.device('cuda')

    def setup_tokenizers(self):
        bert_autotokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
        t5_autotokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.bert_tokenizer = BertTokenizer(self.config, bert_autotokenizer)
        self.flan_t5_tokenizer = FlanT5Tokenizer(self.config, t5_autotokenizer)
        self.config.bert_vocab_size = self.bert_tokenizer.n_vocab
        self.config.t5_vocab_size = self.flan_t5_tokenizer.n_vocab

    def setup_datasets(self):
        self.stage1_train_dataset = Blip2Dataset(self.config, split="train", tokenizer=self.bert_tokenizer.tokenize_text, type="bert")
        self.stage1_train_dataloader = DataLoader(self.stage1_train_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.stage2_train_dataset = Blip2Dataset(self.config, split="train", tokenizer=self.flan_t5_tokenizer.tokenize_text, type="flan_t5")
        self.stage2_train_dataloader = DataLoader(self.stage2_train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def train_stage_1(self, num_epochs=20):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        itc_loss_func = ITCLoss()
        itm_loss_func = ITMLoss(self.config)
        itg_loss_func = ITGLoss(self.bert_tokenizer.pad_token_id)

        itc_loss_func = itc_loss_func.to(self.device)
        itm_loss_func = itm_loss_func.to(self.device)
        itg_loss_func = itg_loss_func.to(self.device)

        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epochs):
            iteration = 0
            print(f"***************   Epoch {epoch + 1}  ***************")
            for img, cls_caption, dec_caption in self.stage1_train_dataloader:
                img = img.to(self.device)
                cls_caption = cls_caption.to(self.device)
                dec_caption = dec_caption.to(self.device)
                B, _, _, _ = img.shape

                itc_query_embds, itc_text_embds, itm_query_embds, itm_text_embds, itg_logits = self.model.stage1(img, cls_caption, dec_caption)

                # ITC Loss
                itc_loss, itc_logits = itc_loss_func(itc_query_embds, itc_text_embds)

                # ITM Loss
                idx = torch.arange(B, device = self.device)
                itc_logits[idx, idx] = -1e9
                next_best_text_value , next_best_text_idx = torch.max(itc_logits,dim=1)
                mismatched_cls_caption = cls_caption[next_best_text_idx]
                mismatched_dec_caption = dec_caption[next_best_text_idx]

                _,_,mismatched_itm_query_embeds,_,_ = self.model.stage1(img, mismatched_cls_caption, mismatched_dec_caption)

                itm_query_embed_concatenated = torch.concat((itm_query_embds, mismatched_itm_query_embeds) ,dim=0 )
                itm_labels = torch.zeros(2 * B, dtype=torch.long).to(self.device)
                itm_labels[B:] = 1
                itm_loss = itm_loss_func(itm_query_embed_concatenated, itm_labels)

                # ITG Loss
                itg_labels = torch.concat((dec_caption[:, 1:], dec_caption[:, -1].unsqueeze(1)), dim=1)
                itg_loss = itg_loss_func(itg_logits, itg_labels)


                total_loss = itc_loss + itm_loss + itg_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if iteration % 10 == 0:
                    print(f"Epoch {epoch + 1} : Iter [{iteration} / {len(self.stage1_train_dataloader)}]")
                    print(f"Total Loss: {total_loss.item()}")
                    print(f"ITC Loss: {itc_loss}, ITM Loss: {itm_loss}, ITG Loss: {itg_loss}")
                    print("" + "*" * 50)
                iteration += 1

            torch.save(self.model.state_dict(), "q_former.pt")


    def train_stage_2(self, num_epochs=20, checkpoint_path="q_former.pt"):
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))

        optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epochs):
            iteration = 0
            print(f"***************   Epoch {epoch + 1}  ***************")
            for img, input_caption, input_mask in self.stage2_train_dataloader:
                img = img.to(self.device)
                input_caption = input_caption.to(self.device).squeeze()
                input_mask = input_mask.to(self.device).squeeze()
                B, S = input_caption.shape

                out = self.model.stage2(img, input_caption[:,:2], input_caption[:,2:], input_mask[:,:2], (B, S))
                total_loss = out.loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if iteration % 10 == 0:
                    print(f"Epoch {epoch + 1} : Iter [{iteration} / {len(self.stage2_train_dataloader)}]")
                    print(f"Total Loss: {total_loss.item()}")
                iteration += 1

            torch.save(self.model.state_dict(), "blip2.pt")

if __name__ == "__main__":
    # wandb.init(project="blip2", entity="vlm")
    # wandb.config.update(config)
    
    blip2_trainer = Blip2Trainer()
    blip2_trainer.train_stage_1(num_epochs=20)
    blip2_trainer.train_stage_2(num_epochs=20, checkpoint_path="q_former.pt")

    # wandb.finish()
