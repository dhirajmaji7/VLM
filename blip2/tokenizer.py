import torch
from config import Blip2Config

class FlanT5Tokenizer:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.context_length = config.context_length
        self.n_vocab = len(tokenizer) 
        self.config.t5_vocab_size = self.n_vocab

    def tokenize_text(self, texts: str):
        batch_encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.context_length
        )
        return batch_encoded["input_ids"].to(torch.long), batch_encoded["attention_mask"].to(torch.long)

        
    def decode(self, token_ids):
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return decoded


class BertTokenizer:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.pad_token = tokenizer.pad_token
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.cls_token = "<cls>"
        self.dec_token = "<dec>"

        additional_tokens = [self.bos_token, self.cls_token, self.dec_token, self.eos_token]
        tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})

        self.pad_token_id = tokenizer.convert_tokens_to_ids(self.pad_token)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(self.eos_token)
        self.bos_token_id = tokenizer.convert_tokens_to_ids(self.bos_token)
        self.cls_token_id = tokenizer.convert_tokens_to_ids(self.cls_token)
        self.dec_token_id = tokenizer.convert_tokens_to_ids(self.dec_token)

        self.context_length = config.context_length
        self.n_vocab = len(tokenizer)
        self.config.bert_vocab_size = len(tokenizer)

    def tokenize_text(self, text: str):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        padded = token_ids + [self.pad_token_id] * (self.context_length - len(token_ids) - 1)
        cls_token_ids = [self.cls_token_id] + padded
        dec_token_ids = [self.dec_token_id] + padded
        return torch.tensor(cls_token_ids, dtype=torch.long), torch.tensor(dec_token_ids, dtype=torch.long)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)