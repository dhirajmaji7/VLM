import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import CLIPConfig

class CLIPImageEncoder(nn.Module):
    def __init__(self, config: CLIPConfig):
        super(CLIPImageEncoder, self).__init__()
        self.config = config
        self.model = timm.create_model(config.image_encoder_name, 
                                       pretrained=config.image_encoder_pretrained)
        self.model.reset_classifier(0) 

    def forward(self, x):
        features = self.model.forward_features(x)  # [B, C, F]
        cls_token = features[:, 0]  # CLS token is at index 0
        return cls_token


class CLIPTextEncoder(nn.Module):
    def __init__(self, config: CLIPConfig, tokenizer):
        super(CLIPTextEncoder, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.token_embedding = nn.Embedding(tokenizer.n_vocab, config.text_embd_dim)
        self.positional_embedding = nn.Parameter(
            torch.empty(config.context_length, config.text_embd_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.text_embd_dim, 
                                       nhead=config.num_heads,
                                       dim_feedforward=config.feedforward_dim,
                                       activation='gelu',
                                       batch_first=True),
            num_layers=config.num_layers
        )
        self.ln_final = nn.LayerNorm(config.text_embd_dim)
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)  # [batch_size, seq_len, embd_dim]
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)  # [batch_size, seq_len, embd_dim]

        # Find the index of EOS in each sequence
        eos_mask = (token_ids == self.tokenizer.eos_token_id)  # [batch_size, seq_len, 1]
        assert eos_mask.any(dim=1).all(), "EOS token not found in every sequence."
        # Get the index of the last EOS token in each sequence
        eos_indices = eos_mask.float().argmax(dim=1)
        
        # Gather the features at the EOS token positions
        batch_size = x.size(0)
        features = x[torch.arange(batch_size), eos_indices]  # [batch_size, embd_dim]
        return features


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig, tokenizer):
        super(CLIPModel, self).__init__()
        self.config = config
        self.image_encoder = CLIPImageEncoder(config)
        self.text_encoder = CLIPTextEncoder(config, tokenizer)
        self.image_projection = nn.Linear(config.image_embd_dim, config.embedding_dim)
        self.text_projection = nn.Linear(config.text_embd_dim, config.embedding_dim)
        self.temperature = nn.Parameter(torch.ones([]) * config.temperature)  # Learnable temperature parameter

    def forward(self, images, token_ids):
        img_embeddings = self.image_encoder(images) # [B, I_f]
        text_embeddings = self.text_encoder(token_ids)    # [B, T_f]
        img_embeddings = self.image_projection(img_embeddings) # [B, T_e]
        text_embeddings = self.text_projection(text_embeddings) # [B, T_e]

        # img_embeddings = F.normalize(img_embeddings, p=2, dim=-1) # [B, T_e]
        # text_embeddings = F.normalize(text_embeddings, p=2, dim=-1) # [B, T_e]
        logits = torch.matmul(img_embeddings, text_embeddings.T) * torch.exp(self.temperature) # [B, B]
        
        return logits
