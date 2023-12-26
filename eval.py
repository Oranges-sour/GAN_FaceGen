from network import GeneratorNet
import torch
import numpy as np
import torchvision

import pygame

device = "cpu"

# 使用pygame之前必须初始化
pygame.init()
# 设置主屏窗口
screen = pygame.display.set_mode((128 * 2, 128 * 2))
screen.fill((0, 0, 0))
pygame.display.set_caption("main")


network = torch.load("model_2.pth", map_location=device)

network.eval()

genn = torch.random.manual_seed(4352)
# x = torch.from_numpy(np.random.randn(1, 32, 32)).to(device=device, dtype=torch.float32)


# target = load_img_tensor("4.jpg")
# target = torch.from_numpy(np.random.randn(1, 32, 32)).to(
#     device=device, dtype=torch.float32
# )
with torch.no_grad():
    while True:
        for event in pygame.event.get():
            # 判断用户是否点了关闭按钮
            if event.type == pygame.QUIT:
                # 卸载所有模块
                pygame.quit()

        noise = torch.normal(
            mean=0,
            std=1,
            size=(16 * 16, 1, 100),
            generator=genn,
            device=device,
            dtype=torch.float32,
        )

        y = network(noise)

        # y = torch.nn.functional.interpolate(y,size=(512,512),mode="bilinear")

        # y = y / 2.0 + 0.5

        y = torchvision.utils.make_grid(y, nrow=16, normalize=True, scale_each=True)
        y = y * 255

        y = y.to(dtype=int).numpy().transpose((2, 1, 0))

        # sur = pygame.surface.Surface(size=(128 * 16, 128 * 16))
        sur = pygame.surfarray.make_surface(array=y)

        pygame.image.save(sur, "124.jpg")
        screen.fill((0, 0, 0))
        screen.blit(sur, dest=(0, 0))
        pygame.display.flip()

        input()
