import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCECriterion(nn.Module):
    """
    InfoNCE loss for CLIP.
    """
    def __init__(self):
        super(InfoNCECriterion, self).__init__()

    def forward(self, logits):
        labels = torch.arange(logits.size(0), device=logits.device)

        # Image-to-Text loss (along rows)
        loss_i = F.cross_entropy(logits, labels)

        # Text-to-Image loss (along columns)
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2