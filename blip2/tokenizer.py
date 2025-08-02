import torch
from config import Blip2Config

class Blip2Tokenizer:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.pad_token = tokenizer.pad_token
        self.eos_token = tokenizer.eos_token
        self.bos_token = "<bos>"
        self.cls_token = "<cls>"
        self.dec_token = "<dec>"

        additional_tokens = [self.bos_token, self.cls_token, self.dec_token]
        tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})

        self.pad_token_id = tokenizer.convert_tokens_to_ids(self.pad_token)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(self.eos_token)
        self.bos_token_id = tokenizer.convert_tokens_to_ids(self.bos_token)
        self.cls_token_id = tokenizer.convert_tokens_to_ids(self.cls_token)
        self.dec_token_id = tokenizer.convert_tokens_to_ids(self.dec_token)

        self.context_length = config.context_length
        self.n_vocab = self.tokenizer.vocab_size

    def tokenize_text(self, text: str, task: str):
        token_ids = self.tokenizer.encode(text)
        token_ids = self.bos_token_id + token_ids + self.eos_token_id
        if task == "cls":
            token_ids = self.cls_token_id + token_ids
        elif task == "dec":
            token_ids = self.dec_token_id + token_ids
        else:
            raise RuntimeError("Task can be cls or dec only.")
        padded = token_ids + [self.pad_token_id] * (self.context_length - len(token_ids))
        return torch.tensor(padded, dtype=torch.long)

    def decode(self, token_ids):
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return decoded