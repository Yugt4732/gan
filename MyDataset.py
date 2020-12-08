from torch.utils.data import dataset, dataloader
from torchvision import transforms, datasets
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import logging
import sys
import os
import torch

device = "cuda"
dataset_path = 'D:/Documents/Py_Docu/data'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))

])
train_data = datasets.MNIST(dataset_path, transform=transform, train=True, download=False)
test_data = datasets.MNIST(dataset_path, transform=transform, train=False, download=False)
train_loader = dataloader.DataLoader(train_data, shuffle=True, batch_size=256)
test_loader = dataloader.DataLoader(test_data, shuffle=True, batch_size=256)

class Lenet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.Relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.Relu(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.Relu(x)
        x = self.fc2(x)
        x = self.Relu(x)
        x = self.fc3(x)

        return x


class Lenet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.Relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.Relu(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.Relu(x)
        x = self.fc2(x)
        x = self.Relu(x)
        x = self.fc3(x)

        return x


class Lenet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.Relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.Relu(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.Relu(x)
        x = self.fc2(x)
        x = self.Relu(x)
        x = self.fc3(x)

        return x


def lenet_eval(net):
    net.eval()
    cnt = 0
    len = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = net(data)
        len += data.shape[0]
        cnt += (torch.argmax(output, dim=1) == target).sum().item()
    accu = cnt/len*100

    return accu

def lenet_model(lenet, epochs, early_stop, id):
    logging.info('----------- Network Initialization --------------')
    # lenet = LeNet().cuda()
    SGD = optim.SGD(lenet.parameters(), lr=0.01, momentum=0.9)
    certion = torch.nn.CrossEntropyLoss()
    max_accu = 0
    early_round = 0
    for epoch in range(1, epochs+1):
        lenet.train()
        early_round += 1
        cnt = 0
        len = 0
        for i, (data, target) in enumerate(train_loader):
            SGD.zero_grad()
            data, target = data.to(device), target.to(device)
            output = lenet(data)
            loss = certion(output, target)
            loss.backward()
            SGD.step()
            len += data.shape[0]
            cnt += (torch.argmax(output, dim=1) == target).sum().item()
        Accu_rate = cnt/len*100
        logging.info("Epoch[{}]: Train accuracy is {:.4f}%".format(epoch, Accu_rate))
        Accu_test = lenet_eval(lenet)
        logging.info("Epoch[{}]: Test accuracy is {:.4f}%".format(epoch, Accu_test))
        if Accu_test > max_accu:
            max_accu = Accu_test
            torch.save(lenet, './LeNet'+str(id)+'.model')
            early_round = 0
        if early_round >= early_stop:
            break
    logging.info('----------- Network training finished --------------')
    logging.info("Max_Accu in test dataset: {:.4f}%".format(max_accu))
    return max_accu

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join('./Lenet_mnist/', 'log_.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

net1 = Lenet1().cuda()
net2 = Lenet2().cuda()
net3 = Lenet3().cuda()
epochs = 1000
early_round = 50

lenet_model(net1, epochs, early_round, 1)
lenet_model(net2, epochs, early_round, 2)
lenet_model(net3, epochs, early_round, 3)