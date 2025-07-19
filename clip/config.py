from dataclasses import dataclass

@dataclass
class CLIPConfig():
    # Model parameters
    image_encoder_name: str = 'vit_tiny_patch16_224'
    image_encoder_pretrained: bool = False
    image_embd_dim: int = 192
    text_embd_dim: int = 512
    embedding_dim: int = 512
    context_length: int = 77
    num_layers: int = 12
    num_heads: int = 8
    feedforward_dim: int = text_embd_dim * 4
    temperature: float = 0.07  # Initial temperature for logits scaling
    
    # Dataset & Tokenizer parameters
    image_dir: str = '../datasets/flickr8k/Images'
    captions_filepath: str = '../datasets/flickr8k/captions.txt'
    image_size: tuple = (224, 224)  # Size to which images will be resized
    random_seed: int = 42
    tokenizer_name: str = 'gpt2'

    # Training parameters
    batch_size: int = 128
    num_workers: int = 4  # Number of workers for data loading
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    runs_dir: str = 'runs'
    checkpoint_dir: str = 'checkpoints'
    checkpoint_interval: int = 10