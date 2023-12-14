import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import cv2

device = "cuda"


class GeneratorNet(nn.Module):
    def __init__(self) -> None:
        super(GeneratorNet, self).__init__()
        self.fc1 = nn.Linear(256, 256, device=device)
        self.fc2 = nn.Linear(256, 512, device=device)
        self.fc3 = nn.Linear(512, 1024, device=device)
        self.fc4 = nn.Linear(1024, 1024 * 3, device=device)

        # self.conv1 = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))

        x = x.reshape(-1, 3, 32, 32)
        # x = torch.nn.functional.interpolate(x, size=(32, 32), mode="bilinear")

        return x


class DiscriminatorNet(nn.Module):
    def __init__(self) -> None:
        super(DiscriminatorNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding=1, device=device
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1, device=device
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 8 * 8, 128, device=device)
        self.fc2 = nn.Linear(128, 1, device=device)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(-1, 16 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
