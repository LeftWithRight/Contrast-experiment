import torchvision
import struct
from torchvision import transforms
from model import Mnist_2NN
import os
from model import *
from torch import optim
import torch.utils.data.dataloader as dataloader
import torch.nn.functional as F
import torch
from train_fc import *
import torch
from split_dataset import *

# 加载MNIST数据集3
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_set = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = dataloader.DataLoader(dataset=test_set, batch_size=128, shuffle=True)
train_dataset = torchvision.datasets.MNIST('data/',train=True, transform=transform, download=True)
train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)


test_txt = open("test_accuracy.txt", mode="a")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net = Mnist_2NN(dropout_prob=0.5).to(dev)
# bathsize = 20
# 定义全局损失函数
# 定义损失函数
loss_func = nn.CrossEntropyLoss()
# 优化算法的，随机梯度下降法
# 使用Adam下降法
opti = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
global_parameters = {}
for key, var in net.state_dict().items():
    global_parameters[key] = var.clone()
  #下面是Iid
    # train_set = train_dataset
    #下面是NOIID
train_set = split_dataset()


for epoch in range(1, 51):
    # 开始训练，
    net.train()
    net = train(train_set, net, epoch, loss_func, opti)

    # 载入测试集
    net.eval()
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
    loss_value = loss_num / len(test_loader)
    print(sum_accu)
    print("\n" + 'accuracy: {}'.format(sum_accu / num))
    print("loss:" + str(loss_value) + "\n")

    test_txt.write(str(epoch) + " ")
    test_txt.write(str(float(sum_accu/num)) + " ")
    test_txt.write(str(loss_value) + "\n")