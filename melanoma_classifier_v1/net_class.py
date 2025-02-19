import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


# 50px x 50px
img_size = 50

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 5)

        # fc input size = 128 x 2 x 2 per shape output of conv3
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 2) # binary classification
    
    def forward(self, x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        # FC input: 128 x 2 x 2
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # output
        x = F.softmax(x)

        return x




#net = Net()
#test_img = torch.randn(img_size, img_size).view(-1, 1, img_size, img_size) # dummy data for test run
#output = net(test_img)