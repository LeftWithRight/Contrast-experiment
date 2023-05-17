import torch.optim as optim
import torchvision
import torch.nn
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import dataloader
from model import Net

train_dataset = torchvision.datasets.MNIST('data/',train=True, transform=transforms.ToTensor(), download=True)
train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_set = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
test_loader = dataloader.DataLoader(dataset=test_set, batch_size=128, shuffle=True)

# 定义神经网络
net = Net()

# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(50):
    for i, data in train_loader:
        # 将梯度归零
        optimizer.zero_grad()

        # 前向传播
        outputs, regularization_loss = net(data)
        loss = criterion(outputs, labels) + 0.001 * regularization_loss  # 将正则化项的损失值加入到总的损失值中

        # 反向传播
        loss.backward()
        optimizer.step()

        # 载入测试集
    sum_accu = 0
    num = 0
    net = net.to(dev)
    loss_num = 0
    for data, label in test_loader:
        data = data.reshape(-1, 28 * 28)
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        loss1 = criterion(preds, label)
        loss_num += loss1.item()
        # print(str(loss1.item))
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean()
        num += 1
    loss_value = loss_num / len(test_loader)
    print("\n" + 'accuracy: {}'.format(sum_accu / num))
    print("loss:" + str(loss_value) + "\n")

