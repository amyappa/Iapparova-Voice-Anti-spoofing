import torch
from torch import nn


class MFM(nn.Module):
    """
    Max-Feature-Map activation.

    Splits the features into two equal groups and takes
    the element-wise maximum between them.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_count = x.shape[1]

        half = feature_count // 2

        first_half = x[:, :half]
        second_half = x[:, half:]

        return torch.maximum(first_half, second_half)


class LCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_reduction = nn.Conv1d(
            in_channels=257,
            out_channels=60,
            kernel_size=1,
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            MFM(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(48),
            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(
                in_channels=48,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=32 * 3 * 46,
                out_features=160,
            ),
            MFM(),
            nn.Dropout(p=0.75),
            nn.BatchNorm1d(80),
            nn.Linear(
                in_features=80,
                out_features=2,
            ),
        )

    def forward(self, data_object, **batch):
        x = data_object.squeeze(1)
        x = self.feature_reduction(x)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        logits = self.classifier(x)

        return {"logits": logits}
