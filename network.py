import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import cv2

device = "cuda"


class GeneratorNet(nn.Module):
    def __init__(self) -> None:
        super(GeneratorNet, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(
            in_channels=100,
            out_channels=1024,
            kernel_size=4,
            stride=2,
            bias=False,
            device=device,
        )
        self.bn1 = nn.BatchNorm2d(num_features=1024, device=device)

        self.tconv2 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            device=device,
        )
        self.bn2 = nn.BatchNorm2d(num_features=512, device=device)

        self.tconv3 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            device=device,
        )
        self.bn3 = nn.BatchNorm2d(num_features=256, device=device)

        self.tconv4 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            device=device,
        )
        self.bn4 = nn.BatchNorm2d(num_features=128, device=device)

        self.tconv5 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device,
        )

    def forward(self, x):
        x = x.reshape(-1, 100, 1, 1)

        x = torch.relu(self.bn1(self.tconv1(x)))
        x = torch.relu(self.bn2(self.tconv2(x)))
        x = torch.relu(self.bn3(self.tconv3(x)))
        x = torch.relu(self.bn4(self.tconv4(x)))
        x = torch.tanh(self.tconv5(x))

        # print(x.shape)
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
