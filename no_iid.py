import torchvision
from torchvision import transforms
from train_fc import *
from model import *
import torch.nn.functional as F
from torch import optim
import torch.utils.data.dataloader as dataloader
import torch
import struct
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from icecream import ic
from PIL import Image
# class torch.utils.data.ConcatDataset


from torch.utils.data import ConcatDataset



train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=False)
train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=20, shuffle=True)

test_set = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=False)
test_loader = dataloader.DataLoader(dataset=test_set, batch_size=10000, shuffle=True)


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
print(flag)

lst0 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst0[i] = label_array[0][i]
train_set_0 = Subset(train_dataset, lst0)
print(len(lst0))


# 获得数据子集
lst1 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst1[i] = label_array[1][i]
train_set_1 = Subset(train_dataset, lst1)


lst2 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst2[i] = label_array[2][i]
train_set_2 = Subset(train_dataset, lst2)

lst3 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst1[i] = label_array[3][i]
train_set_3 = Subset(train_dataset, lst3)

lst4 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst4[i] = label_array[4][i]
train_set_4 = Subset(train_dataset,lst4)

lst5 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst5[i] = label_array[5][i]
train_set_5 = Subset(train_dataset, lst5)

lst6 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst6[i] = label_array[6][i]
train_set_6 = Subset(train_dataset,lst6)

lst7 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst7[i] = label_array[7][i]
train_set_7 = Subset(train_dataset,lst7)

lst8 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst8[i] = label_array[8][i]
train_set_8 = Subset(train_dataset,lst8)


lst9 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst9[i] = label_array[9][i]
train_set_9 = Subset(train_dataset, lst9)


# data = data.squeeze()   # 删除通道维度 [64,1,28,28]->[64,28,28]
train_loader = dataloader.DataLoader(dataset=train_set_9, batch_size=20, shuffle=True)

fig, ax = plt.subplots(
    nrows=3,
    ncols=4,
    sharex=True,
    sharey=True, )

ax = ax.flatten()

for i in range(12):
    img = train_set_8.data[i]
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# fig, ax = plt.subplots(
#     nrows=3,
#     ncols=4,
#     sharex=True,
#     sharey=True, )
#
# ax = ax.flatten()
#
#
#
# for i in range(12):
#     # 只查看了前面12张图片
#     img = train_set_9.data[i]
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#



# ic(len(train_dataset))
# ic(train_dataset[0])
# ic(train_dataset[0][0])

# test_txt = open("test_accuracy.txt", mode="a")
# net = Mnist_2NN()
# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# lr = 0.001
#
# # 定义损失函数
# loss_func = F.cross_entropy
# # 优化算法的，随机梯度下降法
# # 使用Adam下降法
# opti = optim.Adam(net.parameters(), lr=lr)
#
# for epoch in range(1, 51):
#     net = train(train_set_0, net, epoch, loss_func, opti)
#     net = train(train_set_1, net, epoch, loss_func, opti)
#     net = train(train_set_2, net, epoch, loss_func, opti)
#     net = train(train_set_3, net, epoch, loss_func, opti)
#     net = train(train_set_4, net, epoch, loss_func, opti)
#     net = train(train_set_5, net, epoch, loss_func, opti)
#     net = train(train_set_6, net, epoch, loss_func, opti)
#     net = train(train_set_7, net, epoch, loss_func, opti)
#     net = train(train_set_8, net, epoch, loss_func, opti)
#     net = train(train_set_9, net, epoch, loss_func, opti)
#
#     # 载入测试集
#     sum_accu = 0
#     num = 0
#     net = net.to(dev)
#     loss_num = 0
#     for data, label in test_loader:
#         data = data.reshape(-1, 28 * 28)
#         data, label = data.to(dev), label.to(dev)
#         preds = net(data)
#         loss1 = loss_func(preds, label)
#         loss_num += loss1.item()
#         # print(str(loss1.item))
#         preds = torch.argmax(preds, dim=1)
#         sum_accu += (preds == label).float().mean()
#         num += 1
#
#     print("\n" + 'accuracy: {}'.format(sum_accu / num))
#     print("loss:" + str(loss_num) + "\n")
#
#     test_txt.write("communicate round " + str(epoch) + "  ")
#     test_txt.write('accuracy: ' + str(float(sum_accu / num)) + "\n")
#
#     test_txt.write("loss:" + str(loss_num) + "\n")
