import torch
import torch.nn as nn
class abc(nn.Module):
    def __init__(self, c1, c2):
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(c2 * 3, c2, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y = torch.cat((x, x1, x2), dim=1)
        return self.conv3(y)

a = abc()
a.forward()