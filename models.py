import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.models as models
'''
Basic convolutional neural net for images.
'''
class convNN(torch.nn.Module):
    def __init__(self):
        super(convNN, self).__init__()
        # Pooling and convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.sig = nn.Sigmoid()

        # Fully connected layers
        self.fc1 = nn.Linear(35344, 300)
        self.fc2 = nn.Linear(300, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, input):
        x = self.pool(self.sig(self.conv1(input)))
        x = self.pool(self.sig(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        x = self.sig(self.fc3(x))

        return x
    
class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.resnet = models.ResNet(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=6)
    
    def forward(self, x):
        return self.resnet.forward(x)

class resnet34(nn.Module):
    def __init__(self):
        super(resnet34, self).__init__()
        self.resnet = models.ResNet(models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=6)
    
    def forward(self, x):
        return self.resnet.forward(x)

class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.resnet = models.ResNet(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=6)

    def forward(self, x):
        return self.resnet.forward(x)