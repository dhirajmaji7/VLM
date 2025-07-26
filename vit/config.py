from dataclasses import dataclass

@dataclass
class VitConfig:
    embd_dim = 768
    num_layers = 12
    num_heads = 12
    num_classes = 10
    img_size = 256
    patch_size = 16 
    num_channels = 3