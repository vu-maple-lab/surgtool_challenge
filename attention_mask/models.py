import numpy as np
import torch
import torch.nn as nn

class ResNet(nn.Module):
    
    def __init__(self, in_channels=4, num_classes=14):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.last_layer = torch.nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.last_layer(x)
        # x = self.sigmoid(x) 
        return x