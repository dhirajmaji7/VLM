import torch
import tiktoken

class CLIPTextTokenizer:
    def __init__(self, context_length=77):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.bos_token_id = 49406  # <|startoftext|> used in CLIP
        self.eos_token_id = 49407  # <|endoftext|> used in CLIP
        self.n_vocab = self.tokenizer.n_vocab
        self.context_length = context_length
        print(f"Using tokenizer: {self.tokenizer.name} with vocab size: {self.n_vocab}")

    def tokenize_text(self, text):
        token_ids = self.tokenizer.encode(text)
        token_ids = [self.bos_token_id] + token_ids[:self.context_length-2] + [self.eos_token_id]
        padded = token_ids + [0] * (self.context_length - len(token_ids))
        return torch.tensor(padded, dtype=torch.long)