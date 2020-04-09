import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride = 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride = 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride = 2)
        
        self.fc1 = nn.Linear(64*16*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        
        
    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        print(x.size())
        

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.maxpool(x)
        print(x.size())


        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.maxpool(x)
        print(x.size())
        
        x = x.view(-1, 64*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x