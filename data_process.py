# coding:utf8
import os

# 图片路径
path = "data1"
# 目标文件夹路径
path1 = "data1"
filelist = os.listdir(path)

j = 0

for i in filelist:
    # 判断该路径下的文件是否为图片
    if i.endswith(".jpg"):
        # 打开图片
        src = os.path.join(os.path.abspath(path), i)
        # 重命名
        dst = os.path.join(
            os.path.abspath(path1), "b" + f"{j}" + ".jpg"
        )
        # 执行操作
        os.rename(src, dst)
        j += 1
