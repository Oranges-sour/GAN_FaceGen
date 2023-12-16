import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import cv2

device = "cpu"


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

        self.leaky = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=4,
            stride=1,
            padding=1,
            bias=False,
            device=device,
        )

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            device=device,
        )
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            device=device,
        )
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            device=device,
        )
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(
            in_channels=512,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
            device=device,
        )

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.bn2(self.conv2(x)))
        x = self.leaky(self.bn3(self.conv3(x)))
        x = self.leaky(self.bn4(self.conv4(x)))

        # print(x.shape)

        x = torch.sigmoid(self.conv5(x))

        x = x.reshape(-1, 1)

        return x
