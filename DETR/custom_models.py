import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.models as models

import torchvision
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
    
class DETR(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=64, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # Convolutional layers from ResNet-50 model
        model_resnet50 = torchvision.models.resnet50(pretrained = True)
        self.backbone = nn.Sequential(*list(model_resnet50.children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads,
        num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H , W = h.shape[-2:]
        pos = torch.cat([
        self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
        self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
        self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()

    '''
    x = self.backbone(inputs)
    b = inputs.shape[0]
    h = self.conv(x)
    H , W = h.shape[-2:]
    pos = torch.cat([
    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),    # TODO: The trailing one should prob be the batch size
    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
    ], dim=-1).flatten(0, 1).unsqueeze(1).expand(-1, b, -1)
    h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
    self.query_pos.unsqueeze(1).expand(-1, b, -1))
    print(h)
    return self.linear_class(h), self.linear_bbox(h).sigmoid()[0,:,:]
    '''
