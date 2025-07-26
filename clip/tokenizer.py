import torch
import tiktoken

from config import CLIPConfig
class CLIPTextTokenizer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = tiktoken.get_encoding(config.tokenizer_name)

        self.pad_token = "<|pad|>"
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"

        self.special_tokens = {
            self.pad_token: self.tokenizer.n_vocab,
            self.bos_token: self.tokenizer.n_vocab + 1,
            self.eos_token: self.tokenizer.n_vocab + 2
        }

        self.pad_token_id = self.special_tokens[self.pad_token]
        self.bos_token_id = self.special_tokens[self.bos_token]
        self.eos_token_id = self.special_tokens[self.eos_token]

        self.context_length = config.context_length
        self.n_vocab = self.tokenizer.n_vocab + len(self.special_tokens)

    def tokenize_text(self, text: str):
        token_ids = self.tokenizer.encode(text)
        token_ids = [self.bos_token_id] + token_ids[:self.context_length - 2] + [self.eos_token_id]
        padded = token_ids + [self.pad_token_id] * (self.context_length - len(token_ids))
        return torch.tensor(padded, dtype=torch.long)

    def decode(self, token_ids):
        # Remove special tokens before decoding
        cleaned = [tid for tid in token_ids if tid < self.tokenizer.n_vocab]
        return self.tokenizer.decode(cleaned)
