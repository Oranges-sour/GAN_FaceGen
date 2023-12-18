import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random

import cv2

import pygame

# 使用pygame之前必须初始化
pygame.init()
# 设置主屏窗口
screen = pygame.display.set_mode((32 * 2 * 8, 32 * 2 * 8))
screen.fill((0, 0, 0))
pygame.display.set_caption("main")


from network import GeneratorNet
from network import DiscriminatorNet
from network import device

from tools import load_real_img
from tools import batch_size


real_img = load_real_img()

generator_net = GeneratorNet()
discriminator_net = DiscriminatorNet()

optimizer_G = optim.RMSprop(generator_net.parameters(), lr=5e-5)
optimizer_D = optim.RMSprop(discriminator_net.parameters(), lr=5e-5)


print("hi")

ze = torch.zeros((1, 1), requires_grad=False, device=device)
on = torch.ones((1, 1), requires_grad=False, device=device)

for epo in range(0, 60000):
    # 训练分类器
    for i, data in enumerate(real_img, 0):
        a, b = data

        noise = torch.normal(mean=0.0, std=1.0, size=(batch_size, 100), device=device)
        gen_img = generator_net(noise).detach()

        real_score = discriminator_net(a)

        fake_score = discriminator_net(gen_img)

        loss_d = -(torch.mean(real_score) - torch.mean(fake_score))

        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()

        for p in discriminator_net.parameters():
            p.data.clamp_(-0.01, 0.01)

    # 训练生成器
    noise = torch.normal(mean=0.0, std=1.0, size=(batch_size, 100), device=device)

    gen_img = generator_net(noise)
    loss_g = -torch.mean(gen_img.view(-1))

    optimizer_G.zero_grad()
    loss_g.backward()
    optimizer_G.step()

    if epo % 20 == 0:
        screen.fill((0, 0, 0))

        generator_net.eval()
        with torch.no_grad():
            for kkk in range(0, 8):
                for jjj in range(0, 8):
                    rand_gen = torch.Generator(device=device)
                    rand_gen.manual_seed(kkk * 100 + jjj)

                    noise = torch.normal(
                        mean=0.0,
                        std=1.0,
                        size=(1, 100),
                        generator=rand_gen,
                        device=device,
                    )
                    gi = generator_net(noise)
                    # gi = real_img[1]
                    # # print(gi.shape)
                    # gi = gi.reshape(-1, 3, 32, 32)

                    for event in pygame.event.get():
                        # 判断用户是否点了关闭按钮
                        if event.type == pygame.QUIT:
                            # 卸载所有模块
                            pygame.quit()

                    gi = gi / 2 + 0.5
                    gi = gi.to(device="cpu")
                    for i in range(0, 32):
                        for j in range(0, 32):
                            w = 2
                            pygame.draw.rect(
                                screen,
                                color=(
                                    gi[0][2][j][i] * 255,
                                    gi[0][1][j][i] * 255,
                                    gi[0][0][j][i] * 255,
                                ),
                                rect=(
                                    kkk * 32 * w + i * w,
                                    jjj * 32 * w + j * w,
                                    (i + 1) * w,
                                    (j + 1) * w,
                                ),
                            )
                            # pygame.draw.rect(
                            #     screen,
                            #     color=(
                            #         255,
                            #         255,
                            #         255,
                            #     ),
                            #     rect=(i * 6, j * 6, (i + 1) * 6, (j + 1) * 6),
                            # )

        pygame.display.flip()
        generator_net.train()

        pygame.image.save(screen, f"out/{epo}.jpg")
        print(
            f"{epo}:g:{loss_g.item():.4f}  d:{loss_d.item():4f} real_score:{torch.mean(real_score).item():.4f} fake_score:{torch.mean(fake_score).item():.4f}"
        )

# torch.save(generator_net, "model_1.pth")
