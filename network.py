import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import cv2

device = "cpu"


class GeneratorNet(nn.Module):
    def __init__(self) -> None:
        super(GeneratorNet, self).__init__()

        self.conv1 = nn.ConvTranspose2d(
            in_channels=100,
            out_channels=1024,
            kernel_size=4,
            stride=2,
            padding=0,
            device=device,
        )

        self.conv2 = nn.ConvTranspose2d(
            in_channels=1024,
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
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            device=device,
        )

        self.conv5 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            device=device,
        )

        self.conv6 = nn.ConvTranspose2d(
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
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.tanh(self.conv6(x))

        x = x.reshape(-1, 3, 128, 128)

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
            kernel_size=4,
            stride=2,
            padding=1,
            device=device,
        )

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=196,
            kernel_size=4,
            stride=2,
            padding=1,
            device=device,
        )

        self.conv3 = nn.Conv2d(
            in_channels=196,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            device=device,
        )

        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=4,
            stride=4,
            padding=1,
            device=device,
        )

        self.conv5 = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            device=device,
        )

    def forward(self, x):
        x = x.reshape(-1, 3, 128, 128)

        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        x = self.leaky(self.conv3(x))
        x = self.leaky(self.conv4(x))
        x = self.conv5(x)

        x = x.reshape(-1, 1)

        return x
