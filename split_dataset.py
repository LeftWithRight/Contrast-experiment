import torchvision
import struct
from torchvision import transforms
from torch.utils.data import Subset
# from train_fc import *
# from Block import Block
# from model import *
# import os
# import torch.nn.functional as F
# from torch import optim
import numpy as np
import torch.utils.data.dataloader as dataloader
from torch.utils.data import ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
# torch.utils.data.random_split

def split_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)

    # 读取标签数据集
    with open('./train-labels.idx1-ubyte', 'rb') as lbpath:
        labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    # 存放标签的二维列表
    label_array = []                  # 创建一个空列表
    for i in range(10):      # 创建一个5行的列表（行）
        label_array.append([])        # 在空的列表中添加空的列表
        for j in range(7000):  # 循环每一行的每一个元素（列）
            label_array[i].append(j)  # 为内层列表添加元素
    for i in range(10):
        for j in range(7000):
            label_array[i][j] = 0

    #获得下标
    flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(60000):
        value = labels[i]
        flag[value] = flag[value] + 1
        j = flag[value]
        label_array[value][j] = i
    # print('the ', i, 'of picture is ', label_array[value][flag[value]])



    lst0 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst0[i] = label_array[0][i]
    train_set_01 = Subset(train_dataset, lst0)
    train_set_02 = Subset(train_dataset, range(3000))
    train_set_0_ls = [train_set_01, train_set_02]
    train_set_0 = ConcatDataset(train_set_0_ls)

    lst1 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst1[i] = label_array[1][i]
    train_set_11 = Subset(train_dataset, lst1)
    train_set_12 = Subset(train_dataset, range(3000,6000))
    train_set_1_ls = [train_set_11, train_set_12]
    train_set_1 = ConcatDataset(train_set_1_ls)


    lst2 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst2[i] = label_array[2][i]
    train_set_21 = Subset(train_dataset, lst2)
    train_set_22 = Subset(train_dataset, range(6000,9000))
    train_set_2_ls = [train_set_21, train_set_22]
    train_set_2 = ConcatDataset(train_set_2_ls)

    lst3 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst3[i] = label_array[3][i]
    train_set_31 = Subset(train_dataset, lst3)
    train_set_32 = Subset(train_dataset, range(9000,12000))
    train_set_3_ls = [train_set_31, train_set_32]
    train_set_3 = ConcatDataset(train_set_3_ls)

    lst4 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst4[i] = label_array[4][i]
    train_set_41 = Subset(train_dataset, lst4)
    train_set_42 = Subset(train_dataset, range(12000,15000))
    train_set_4_ls = [train_set_41, train_set_42]
    train_set_4 = ConcatDataset(train_set_4_ls)

    lst5 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst5[i] = label_array[5][i]
    train_set_51 = Subset(train_dataset, lst5)
    train_set_52 = Subset(train_dataset, range(15000,18000))
    train_set_5_ls = [train_set_51, train_set_52]
    train_set_5 = ConcatDataset(train_set_5_ls)

    lst6 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst6[i] = label_array[6][i]
    train_set_61 = Subset(train_dataset, lst6)
    train_set_62 = Subset(train_dataset, range(18000,21000))
    train_set_6_ls = [train_set_61, train_set_62]
    train_set_6 = ConcatDataset(train_set_6_ls)

    lst7 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst7[i] = label_array[7][i]
    train_set_71 = Subset(train_dataset, lst7)
    train_set_72 = Subset(train_dataset, range(21000,24000))
    train_set_7_ls = [train_set_71, train_set_72]
    train_set_7 = ConcatDataset(train_set_7_ls)

    lst8 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst8[i] = label_array[8][i]
    train_set_81 = Subset(train_dataset, lst8)
    train_set_82 = Subset(train_dataset, range(24000,27000))
    train_set_8_ls = [train_set_81, train_set_82]
    train_set_8 = ConcatDataset(train_set_8_ls)

    lst9 = list(map(lambda x: 0, range(3000)))
    for i in range(3000):
        lst9[i] = label_array[9][i]
    train_set_91 = Subset(train_dataset, lst9)
    train_set_92 = Subset(train_dataset, range(27000,30000))
    train_set_9_ls = [train_set_91, train_set_92]
    train_set_9 = ConcatDataset(train_set_9_ls)

    train_set_full_ls = [train_set_0, train_set_1,train_set_2,train_set_3,train_set_4,train_set_5,train_set_6,train_set_7,train_set_8,train_set_9]
    train_set_full = ConcatDataset(train_set_full_ls)
    print("使用NoIid数据集开始训练")
    return train_set_full


if __name__ == "__main__":
    train_set = split_dataset()
    print(len(train_set))

    # for i in range(10):
    #     data, label = train_set_9[i]
    #     print(label)
    #     img = np.array(data + 1) * 127.5
    #     img = np.reshape(img, [28, 28]).astype(np.uint8)
    #     plt.imshow(img, 'gray')
    #     plt.show()




# train_set_0 = Subset(train_dataset,range(6000))
# train_set_1 = Subset(train_dataset, range(6000,12000))
# train_set_2 = Subset9(train_dataset, range(12000,18000))
# train_set_3 = Subset(train_dataset, range(18000,24000))
# train_set_4 = Subset(train_dataset, range(24000,30000))
# train_set_5 = Subset(train_dataset, range(30000, 36000))
# train_set_6 = Subset(train_dataset,range(36000,420000))
# train_set_7 = Subset(train_dataset,range(42000,48000))
# train_set_8 = Subset(train_dataset,range(48000,54000))
# train_set_9 = Subset(train_dataset,range(52000,60000))