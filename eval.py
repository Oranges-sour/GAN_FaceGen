from network import GeneratorNet
from network import load_img_tensor
from network import device
import torch
import numpy as np


import pygame

# 使用pygame之前必须初始化
pygame.init()
# 设置主屏窗口
screen = pygame.display.set_mode((32 * 4, 32 * 4))
screen.fill((0, 0, 0))
pygame.display.set_caption("main")


network = torch.load("model_1.pth", map_location=device)

network.eval()

x = load_img_tensor("4.jpg")

genn = torch.random.manual_seed(114514)
# x = torch.from_numpy(np.random.randn(1, 32, 32)).to(device=device, dtype=torch.float32)
noise = torch.normal(
    mean=0, std=1, size=(1, 32), generator=genn, device=device, dtype=torch.float32
)

# target = load_img_tensor("4.jpg")
# target = torch.from_numpy(np.random.randn(1, 32, 32)).to(
#     device=device, dtype=torch.float32
# )


with torch.no_grad():
    y = network(x, noise)

    for event in pygame.event.get():
        # 判断用户是否点了关闭按钮
        if event.type == pygame.QUIT:
            # 卸载所有模块
            pygame.quit()

    screen.fill((0, 0, 0))
    w = 4
    for i in range(0, 32):
        for j in range(0, 32):
            k = y[0][0][j][i]
            # print(k)
            k = min(max(k, 0.0), 1.0)
            pygame.draw.rect(
                screen,
                (255 * k, 255 * k, 255 * k),
                (i * w, j * w, i * w + w, j * w + w),
            )
    pygame.display.flip()

input()
