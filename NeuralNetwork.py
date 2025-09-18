import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        return F.relu(self.conv2(x))

    def backward(self, x):
            print("hello")



