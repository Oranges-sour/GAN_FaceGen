import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import cv2

device = "cuda"


class GeneratorNet(nn.Module):
    def __init__(self) -> None:
        super(GeneratorNet, self).__init__()

        self.conv1 = nn.ConvTranspose2d(
            in_channels=100,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=0,
            device=device,
        )

        self.conv2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            device=device,
        )

        self.conv3 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            device=device,
        )

        self.conv4 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            device=device,
        )

        self.conv5 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
            device=device,
        )

    def forward(self, x):
        x = x.reshape(-1, 100, 1, 1)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = torch.tanh(self.conv5(x))

        x = x.reshape(-1, 3, 64, 64)

        # print(x.shape)
        # x = torch.nn.functional.interpolate(x, size=(32, 32), mode="bilinear")

        return x


class DiscriminatorNet(nn.Module):
    def __init__(self) -> None:
        super(DiscriminatorNet, self).__init__()

        self.leaky = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            device=device,
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=196,
            kernel_size=3,
            stride=1,
            padding=1,
            device=device,
        )

        self.conv3 = nn.Conv2d(
            in_channels=196,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            device=device,
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 8 * 8, 256, device=device)
        self.fc2 = nn.Linear(256, 1, device=device)

    def forward(self, x):
        x = x.reshape(-1, 3, 64, 64)

        x = self.leaky(self.conv1(x))
        x = self.pool1(x)
        x = self.leaky(self.conv2(x))
        x = self.leaky(self.conv3(x))
        x = self.pool3(x)

        # print(x.shape)

        x = x.reshape(-1, 256 * 8 * 8)

        x = self.leaky(self.fc1(x))
        x = self.fc2(x)

        x = x.reshape(-1, 1)

        return x
