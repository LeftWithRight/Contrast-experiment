import torch
import torch.utils.data.dataloader as dataloader


def train(train_set, net, epoch, lossFun, opti):
    train_loader = dataloader.DataLoader(dataset=train_set, batch_size=20, shuffle=True)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for data, label in train_loader:
        # 加载到GPU上
        data = data.reshape(-1, 28 * 28)
        data, label = data.to(dev), label.to(dev)
        # 模型上传入数据
        net = net.to(dev)
        preds = net(data)
        # 计算损失函数
        '''
            这里应该记录一下模型得损失值 写入到一个txt文件中
        '''
        loss = lossFun(preds, label)
        # 反向传播
        loss.backward()
        # 计算梯度，并更新梯度
        opti.step()
        # 将梯度归零，初始化梯度
        opti.zero_grad()
    # params = net.state_dict()
    print(epoch)
    return net

