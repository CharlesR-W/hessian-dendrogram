"""LeNet-tiny: ~6K param ConvNet for MNIST, no BatchNorm."""

import torch.nn as nn


class LeNetTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5),     # 8*1*25 + 8 = 208
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # -> 8x12x12
            nn.Conv2d(8, 16, 5),    # 16*8*25 + 16 = 3216
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # -> 16x4x4
        )
        self.classifier = nn.Linear(256, 10)  # 256*10 + 10 = 2570

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
