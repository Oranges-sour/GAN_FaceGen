# coding:utf8
import os
import cv2

# 图片路径
path = "data"
# 目标文件夹路径
path1 = "data"
filelist = os.listdir(path)

j = 0

for i in filelist:
    # 判断该路径下的文件是否为图片
    if i.endswith(".png"):
        # 打开图片
        src = os.path.join(os.path.abspath(path), i)

        img = cv2.imread(src,cv2.IMREAD_COLOR)
        cv2.imwrite(f"data/b{j}.jpg", img)

        j += 1
