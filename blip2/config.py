from dataclasses import dataclass

@dataclass
class Blip2Config:
    # Dataset & Tokenizer parameters
    image_dir: str = '../datasets/flickr8k/Images'
    captions_filepath: str = '../datasets/flickr8k/captions.txt'
    image_size: tuple = (224, 224)  # Size to which images will be resized
    random_seed: int = 42
    tokenizer_name: str = 'gpt2'
    context_length: int = 77
    vocab_size: int = 30000 # updated later from tokenizer

    batch_size: int = 16
    num_queries: int = 32
    img_embd_dim: int = 192
    embedding_dim: int = 768
    lm_embedding_dim: int = 512
    num_heads: int = 12

