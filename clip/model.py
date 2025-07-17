import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
class CLIPModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, image_dim, caption_dim, embedding_dim=512):
        super(CLIPModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.image_projection = nn.Linear(image_dim, embedding_dim)
        self.caption_projection = nn.Linear(caption_dim, embedding_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)  # Learnable temperature parameter

    def forward(self, images, token_ids):
        image_features = self.image_encoder(images) # B,I_f
        text_features = self.text_encoder(token_ids)    # B,T_f
        img_embeddings = self.image_projection(image_features) # B,T_e
        text_embeddings = self.caption_projection(text_features) # B,T_e

        img_embeddings = F.normalize(img_embeddings, p=2, dim=-1) # B,T_e
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1) # B,T_e

        logits = torch.matmul(img_embeddings, text_embeddings.t())* torch.exp(self.temperature) # B,B

        return logits


class CLIPImageEncoder(nn.Module):
    def __init__(self):
        super(CLIPImageEncoder, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.model.reset_classifier(0) 

    def forward(self, x):
        features = self.model.forward_features(x)  # shape: (B, num_tokens, dim)
        cls_token = features[:, 0]  # CLS token is at index 0
        return cls_token


class CLIPTextEncoder(nn.Module):
    def __init__(self, vocab_size, context_length=77, width=512, layers=12, heads=8):
        super(CLIPTextEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=width, nhead=heads),
            num_layers=layers
        )
        self.ln_final = nn.LayerNorm(width)
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, token_ids):
        # token_ids: [batch_size, seq_len]
        x = self.token_embedding(token_ids)  # [batch_size, seq_len, width]
        x = x + self.positional_embedding[:x.size(1), :]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, width]
        x = self.transformer(x)  # [seq_len, batch_size, width]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, width]
        x = self.ln_final(x)

        # Take the final [EOS] token embedding (assumes EOS token is at end)
        eos_embeddings = x[torch.arange(x.shape[0]), token_ids.argmax(dim=-1)]  # [batch_size, width]

        return eos_embeddings  # [batch_size, width]


class CLIP(nn.Module):
    def __init__(self, vocab_size, image_dim=768, caption_dim=512, embedding_dim=512):
        super(CLIP, self).__init__()
        self.text_encoder = CLIPTextEncoder(vocab_size=vocab_size)
        self.image_encoder = CLIPImageEncoder()
        self.model = CLIPModel(self.text_encoder, self.image_encoder, image_dim, caption_dim, embedding_dim)

    def forward(self, images, texts):
        return self.model(images, texts)