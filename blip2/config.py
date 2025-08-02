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

    batch_size: int = 128
    