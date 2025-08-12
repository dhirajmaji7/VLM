import torch
import torch.nn as nn
import torch.nn.functional as F

class ITCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_embds, text_embds):
        # query_embds: B, 32, d
        # text_embds: B, 77, d
        text_logit = text_embds[:, :1] # B, 1, d
        B, _, _ = text_logit.shape
        B, Qs, d = query_embds.shape 
        query_embds = query_embds.reshape(B * Qs, d)
        text_embds = text_logit.squeeze()
        logits = query_embds @ text_embds.T   # B*Qs,B
        logits = torch.max(logits.reshape(B,Qs,B),dim=1)[0] # B,B
        label = torch.arange(B,device=query_embds.device)
        return (F.cross_entropy(logits,label)+ F.cross_entropy(logits.T,label)) / 2, logits


class ITMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.embedding_dim
        self.classification_layer = nn.Linear(d,2)
    
    def forward(self, query_embd, label):
        # query_embd --> (B,32,768)
        #label ->(B,1) B x [0/1]

        match_logit = self.classification_layer(query_embd) #(B,32,2)
        match_logit = match_logit.mean(dim=1)
        return F.cross_entropy(match_logit,label)


class ITGLoss(nn.Module):
    def __init__(self, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id

    def forward(self, itg_logits, label_token):
        #itg_logits -> B,S,vocab size
        #label_token -> B,S
        B, S, V = itg_logits.shape
        loss = F.cross_entropy(
            itg_logits.view(B * S, V),
            label_token.view(B * S),
            ignore_index=self.pad_token_id
        )
        return loss
