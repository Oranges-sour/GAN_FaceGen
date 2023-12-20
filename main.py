import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.utils as vutils

from torch.utils.tensorboard.writer import SummaryWriter

import math

import random

import cv2

import time


from network import GeneratorNet
from network import DiscriminatorNet
from network import device

from tools import load_real_img
from tools import batch_size

from torchsummary import summary


real_img_data = load_real_img()

generator_net = GeneratorNet()
discriminator_net = DiscriminatorNet()

summary(generator_net, input_size=(batch_size, 1, 100))
summary(discriminator_net, input_size=(batch_size, 3, 64, 64))

# exit(0)

# 日志目录
log_dir_num = f"{int(time.time())}"
print(f"logs/{log_dir_num}")
writer = SummaryWriter(f"logs/{log_dir_num}")


optimizer_G = optim.Adam(generator_net.parameters(), lr=5e-4, betas=(0.5, 0.9))
optimizer_D = optim.Adam(discriminator_net.parameters(), lr=5e-4, betas=(0.5, 0.9))


for epo in range(0, 12000):
    # 训练分类器
    for i, data in enumerate(real_img_data, 0):
        real_img = data

        noise = torch.normal(mean=0.0, std=1.0, size=(batch_size, 100), device=device)
        gen_img = generator_net(noise).detach()

        real_score = discriminator_net(real_img)

        fake_score = discriminator_net(gen_img)

        #
        alpha = torch.rand(size=(batch_size, 1, 1, 1), device=device)
        interpolates = (real_img * alpha + (1 - alpha) * gen_img).requires_grad_(True)
        d_inter = discriminator_net(interpolates)

        # print(d_inter.shape)
        # print(interpolates.shape)

        gradients = torch.autograd.grad(
            outputs=d_inter,
            inputs=interpolates,
            grad_outputs=torch.ones(size=d_inter.size(), device=device),
            retain_graph=True,
            create_graph=True,
        )[0]

        # print(gradients.shape)
        gradients = gradients.reshape(batch_size, -1)
        # print(gradients.shape)
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        #
        loss_d = -(
            torch.mean(real_score)
            - torch.mean(fake_score)
            - 10 * torch.mean(gradient_penalty)
        )

        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()

        # for p in discriminator_net.parameters():
        #     p.data.clamp_(-0.01, 0.01)

    # 训练生成器
    noise = torch.normal(mean=0.0, std=1.0, size=(batch_size, 100), device=device)

    gen_img = generator_net(noise)
    loss_g = -torch.mean(discriminator_net(gen_img))

    optimizer_G.zero_grad()
    loss_g.backward()
    optimizer_G.step()

    writer.add_scalars(
        "loss", {"generator": loss_g.item(), "critic": loss_d.item()}, epo
    )
    writer.add_scalar(
        "w_distance",
        abs(torch.mean(real_score).item() - torch.mean(fake_score).item()),
        epo,
    )

    if epo % 50 == 0:
        # screen.fill((0, 0, 0))
        rand_gen = torch.Generator(device=device)
        rand_gen.manual_seed(114514)
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(64, 1, 100),
            generator=rand_gen,
            requires_grad=False,
            device=device,
        )

        generator_net.eval()
        with torch.no_grad():
            test_gen_image = generator_net(noise)
        generator_net.train()

        tens = vutils.make_grid(test_gen_image, normalize=True, scale_each=True)

        # tt = torch.zeros(size=(3, 64, 64), device=device)
        # tt[1] = 1.0
        writer.add_image("test_generate", tens, epo, dataformats="CHW")

        print(f"{epo}:g:{loss_g.item():.4f}  d:{loss_d.item():.4f}")

# torch.save(generator_net, "model_1.pth")
