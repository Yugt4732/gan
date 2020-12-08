import torch.nn as nn
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.datasets as dst

dataset_path = 'D:/Documents/Py_Docu/data'
# dataset_path = './data'
device = "cuda"

def to_img(x):
    out = 0.5*(x+1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

class discriminator_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1, 64, 3, 1, padding=1)
        self.bn1=nn.BatchNorm2d(64)
        # self.pool1=nn.AvgPool2d(2)
        # nn.Linear(784, 256),
        self.relu=    nn.ReLU(True)
        self.conv2=    nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # nn.AvgPool2d(2),
        # nn.Linear(256, 256),
        # nn.ReLU(True),
        self.conv3=    nn.Conv2d(128, 64, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4= nn.Conv2d(64, 1, 28, 1, 0)
        # self.pool2=    nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # self.fc = nn.Linear(512, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print("get: ",x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.sig(x)
        return x


# class discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 128)
#         self.relu = nn.ReLU(True)
#         self.fc2 = nn.Linear(128,1)
#         self.Sig = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.Sig(x)
#
#         return x


class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(4, 64, 28, 1, 0)
        self.bn1 = nn.BatchNorm2d(64)
        # nn.Linear(input_size, 256),
        self.relu = nn.ReLU(True)
        self.conv2 = nn.ConvTranspose2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, 3, 1, 1)
        self.conv4 = nn.ConvTranspose2d(64, 1, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # self.pool = nn.AvgPool2d(2)
        # view(-1, 588),
        # self.fc = nn.Linear(12544, 784)
        # view(-1, 1024*3),
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # nn.Linear(input_size, 256),
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # nn.Linear(256, 256),
        # nn.ReLU(True)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        # x = self.pool(x)
        # print(x.shape)

        # x = x.view(-1, 12544)
        # x = self.fc(x)
        x = self.tanh(x)

        return x


##########对抗网络、判别器初始化
D = discriminator_cnn()
G = generator()
##
# G=generator(28)

# D.load_state_dict(torch.load('./discriminator.pth'))
D = D.cuda()
# G.load_state_dict(torch.load('./generator.pth'))
G = G.cuda()


#############网络超参数
batch_size = 64
num_epoch = 1000
z_dimension = 4
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
# d_optimizer = torch.optim.SGD(D.parameters(), lr=0.0003)
# g_optimizer = torch.optim.SGD(G.parameters(), lr=0.0003)


########数据集 初始化
img_trainsform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
dataset = datasets.MNIST(dataset_path, train=True, transform=img_trainsform, download=False)
# dataset = dst.CIFAR100('D:/Documents/Py_Docu/data/cifar-100-python',
#                         transform=img_trainsform,
#                         train=True, download=True)
dataloader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size=batch_size,
    shuffle=True
)


##############对抗训练
for epoch in range(num_epoch):
    G_loss = 0
    D_loss = 0
    G_accu = 0
    D_accu = 0
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # img = img.view(num_img, -1)
        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros((num_img))).cuda()
        # real_img = img.cuda()
        # real_label = torch.ones(num_img).cuda()
        # fake_label = torch.zeros((num_img)).cuda()

        ###############training D###############
            ###real_img
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_score = real_out
            ###生成噪声
        z = Variable(torch.randn(num_img, z_dimension, 1, 1)).cuda()
        fake_img = G(z)
        # fake_img = fake_img.view(num_img, -1)
        # print(fake_img.shape)
        # fake_img = to_img(fake_img)
        # print(fake_img.shape)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_score = fake_out

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        ##############trianing G##############
        z = Variable(torch.randn(num_img, z_dimension, 1, 1)).cuda()
        fake_img = G(z)
        # fake_img = to_img(fake_img)
        # fake_img = fake_img.view(num_img, -1)
        output = D(fake_img)
        g_loss = criterion(output, real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % (len(dataloader)-1) == 0:
            print('Epoch[{}/{}], d_loss:{:.6f}, g_loss:{:.6f}'
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                ##???????      d_loss.data.item()
                epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                real_score.data.mean(), fake_score.data.mean()
            ))
        if epoch ==0:
            ## ?????????    real_img.cpu().data
            real_imgs = to_img(real_img.cpu().data)
            save_image(real_imgs, './img_2/real_imgs.png')
    fake_imgs = to_img(fake_img.cpu().data)
    save_image(fake_imgs, './img_2/fake_imgs-{}.png'.format(epoch+1))
    # torch.save(D.state_dict(), './discriminator.pth')

# torch.save(G.state_dict(), './generator.pth')
# print(G.state_dict())

