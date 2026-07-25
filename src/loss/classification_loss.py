import torch
from torch import nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        return {
            "loss": self.loss(logits, labels),
        }
