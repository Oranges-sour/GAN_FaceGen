import torch
import numpy as np
import cv2

from network import device

from torch.utils.data import DataLoader, Dataset


batch_size = 1


# 载入自己的样本数据
class MyDataSet(Dataset):
    def __init__(self):
        self.sample = []
        for i in range(0, 1):
            file = f"data/a{i}.jpg"
            img = cv2.imread(file)
            img = img.transpose((2, 0, 1))

            img = img / 255.0

            img -= 0.5

            img *= 2

            tens = torch.from_numpy(img).to(device=device, dtype=torch.float32)
            self.sample.append((tens, torch.ones((1), device=device)))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        sample = self.sample[index][0], self.sample[index][1]

        return sample


def load_real_img():
    trainset = MyDataSet()
    # 创建训练集数据加载器
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return trainloader
