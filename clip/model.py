import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, image_dim, caption_dim, embedding_dim=512):
        super(CLIPModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.image_projection = nn.Linear(image_dim, embedding_dim)
        self.caption_projection = nn.Linear(caption_dim, embedding_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)  # Learnable temperature parameter

    def forward(self, images, texts):
        image_features = self.image_encoder(images) # B,I_f
        text_features = self.text_encoder(texts)    # B,T_f
        img_embeddings = self.image_projection(image_features) # B,T_e
        text_embeddings = self.caption_projection(text_features) # B,T_e

        image_features = F.normalize(img_embeddings, p=2, dim=-1) # B,T_e
        text_features = F.normalize(text_embeddings, p=2, dim=-1) # B,T_e

        cosine_similarity = torch.matmul(image_features, text_features.t())* torch.exp(self.temperature) # B,B

        return cosine_similarity