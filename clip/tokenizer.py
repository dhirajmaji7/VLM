import torch
import tiktoken

from config import CLIPConfig
class CLIPTextTokenizer:
    def __init__(self, config: CLIPConfig):
        self.config = config
        self.tokenizer = tiktoken.get_encoding(config.tokenizer_name)
        self.bos_token_id = config.bos_token_id 
        self.eos_token_id = config.eos_token_id
        self.context_length = config.context_length
        self.n_vocab = self.tokenizer.n_vocab

    def tokenize_text(self, text):
        token_ids = self.tokenizer.encode(text)
        token_ids = [self.bos_token_id] + token_ids[:self.context_length - 2] + [self.eos_token_id]
        padded = token_ids + [0] * (self.context_length - len(token_ids))
        return torch.tensor(padded, dtype=torch.long)