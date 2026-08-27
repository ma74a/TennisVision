import torch.nn as nn
from torchvision import models


class TennisCourtModel(nn.Module):

    def __init__(self, output_points):
        super().__init__()

        self.resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features,
            out_features=output_points
        )

    def forward(self, x):
        x = self.resnet(x)
        x = x.reshape(-1, 4, 2)

        return x