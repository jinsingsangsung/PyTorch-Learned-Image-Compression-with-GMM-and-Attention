#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
import torch.nn.functional as F

class Hyper_analysis(nn.Module):
    def __init__(self, num_filters=128):
        super(Hyper_analysis, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
        self.leaky_relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu4 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        h, w = x.shape[-2:]
        x = self.leaky_relu2(self.conv2(x))
        pad_list = [(0, w%2, 0, h%2)]
        x = F.pad(x, (0, w%2, 0, h%2))
        x = self.leaky_relu3(self.conv3(x))
        h, w = x.shape[-2:]
        pad_list.append((0, w%2, 0, h%2))
        x = self.leaky_relu4(self.conv4(x))
        x = self.conv5(x)
        return x, pad_list

if __name__ == "__main__":
    hyper_analysis = Hyper_analysis()
    input_image = torch.zeros([1,128,16,16])
    result = hyper_analysis(input_image)
    print(result.shape)
