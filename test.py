import torchvision
from torchvision import datasets, transforms
from train_fc import *
from model import *
import torch.nn.functional as F
from torch import optim
import torch.utils.data.dataloader as dataloader
import torch
from torch.utils.data import ConcatDataset
import struct
from torch.utils.data import Subset
import numpy as np

transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
# #data/表示下载数据集到的目录，transformation表示对数据集进行的相关处理
train_dataset = datasets.MNIST('data/',train=True, transform=transformation,download=True)
test_dataset = datasets.MNIST('data/', train=False, transform=transformation,download=True)


#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)


# print(train_loader)
# print(train_loader)

# 读取标签数据集
with open('./data/MNIST/raw/train-labels-idx1-ubyte', 'rb') as lbpath:
    labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)

# 读取图片数据集
with open('./train-images.idx3-ubyte', 'rb') as imgpath:
    images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols)


label_array = []                  # 创建一个空列表
for i in range(10):      # 创建一个5行的列表（行）
    label_array.append([])        # 在空的列表中添加空的列表
    for j in range(7000):  # 循环每一行的每一个元素（列）
        label_array[i].append(j)  # 为内层列表添加元素
for i in range(10):
    for j in range(7000):
        label_array[i][j] = 0


flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(60000):
    value = labels[i]
    print("the value is ", value)
    flag[value] = flag[value] + 1
    print("the flag value is ", flag[value])
    label_array[value][flag[value]] = i
    # print('the ', i, 'of picture is ', label_array[value][flag[value]])
print(label_array)
print(flag)

lst0 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst0[i] = label_array[0][i]
train_set_0=Subset(train_dataset,lst0)

lst1 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst1[i] = label_array[1][i]
train_set_1=Subset(train_dataset,lst1)

lst2 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst2[i] = label_array[2][i]
train_set_2=Subset(train_dataset,lst2)

lst3 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst1[i] = label_array[3][i]
train_set_3=Subset(train_dataset,lst3)

lst4 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst4[i] = label_array[4][i]
train_set_4=Subset(train_dataset,lst4)

lst5 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst5[i] = label_array[5][i]
train_set_5=Subset(train_dataset, lst5)

lst6 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst6[i] = label_array[6][i]
train_set_6=Subset(train_dataset,lst6)

lst7 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst7[i] = label_array[7][i]
train_set_7=Subset(train_dataset,lst7)

lst8 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst8[i] = label_array[8][i]
train_set_8=Subset(train_dataset,lst8)


lst9 = list(map(lambda x: 0, range(6000)))
for i in range(6000):
    lst1[9] = label_array[9][i]
train_set_9=Subset(train_dataset, lst9)


data_set_list = [train_set_0,train_set_1,train_set_2,train_set_3,train_set_4,train_set_5,train_set_6,train_set_7,train_set_8,train_set_9]

datasets = ConcatDataset(data_set_list)
train_loader = dataloader.DataLoader(dataset=datasets, batch_size=20, shuffle=True)

test_set = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=False)
test_loader = dataloader.DataLoader(dataset=test_set, batch_size=10000, shuffle=True)

net = Mnist_2NN()
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
lr = 0.001

# 定义损失函数
loss_func = F.cross_entropy
# 优化算法的，随机梯度下降法
# 使用Adam下降法
opti = optim.Adam(net.parameters(), lr=lr)
for epoch in range(1, 51):
    net = train(train_dataset, net, epoch, loss_func, opti)

    # 载入测试集
    sum_accu = 0
    num = 0
    net = net.to(dev)
    loss_num = 0
    for data, label in test_loader:
        data = data.reshape(-1, 28 * 28)
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        loss1 = loss_func(preds, label)
        loss_num += loss1.item()
        # print(str(loss1.item))
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean()
        num += 1
        print(num)

    print("\n" + 'accuracy: {}'.format(sum_accu / num))
    print("loss:" + str(loss_num) + "\n")

