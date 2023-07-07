import torchvision.models as models
import torch.nn as nn
import torch
import config

device = torch.device('cuda:0')

class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=1)
        self.net.features[0] = nn.Conv2d(1,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Vgg16_bn(nn.Module):
    def __init__(self,n_per_unit):
        super().__init__()
        self.net = models.vgg16_bn()
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=config.n_class)
        self.net.features[0] = nn.Conv2d(n_per_unit,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Vgg19_bn(nn.Module):
    def __init__(self,n_per_unit):
        super().__init__()
        self.net = models.vgg19_bn()
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=config.n_class)
        self.net.features[0] = nn.Conv2d(n_per_unit,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet34()
        self.net.conv1 = nn.Conv2d(1,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=512,out_features=config.n_class)

    def forward(self,x):
        return self.net(x)

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18()
        self.net.conv1 = nn.Conv2d(1,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=512,out_features=config.n_class)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.net(x)
        x = self.softmax(x)
        return x

def make_model(name,n_per_unit):
    if name == 'Vgg16':
        net = Vgg16()
    elif name == 'Vgg16_bn':
        net = Vgg16_bn(n_per_unit)
    elif name == 'Vgg19':
        net = Vgg19()
    elif name == 'Vgg19_bn':
        net = Vgg19_bn(n_per_unit)
    elif name == 'ResNet18':
        net = Resnet18().to(device)
    elif name == 'ResNet34':
        net = Resnet34().to(device)
    return net