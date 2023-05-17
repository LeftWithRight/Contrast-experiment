import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# 定义神经网络
class Mnist_2NN(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = self.dropout1(tensor)
        tensor = F.relu(self.fc2(tensor))
        tensor = self.dropout2(tensor)
        tensor = self.fc3(tensor)
        return tensor


# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = datasets.MNIST('data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)


# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Mnist_2NN(dropout_prob=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
test_txt = open("test_accuracy.txt", mode="a")


for epoch in range(50):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.view(-1, 784))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs.view(-1, 784))
            test_loss += criterion(outputs, labels).item()

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    test_txt.write(str(test_loss) + "\n")