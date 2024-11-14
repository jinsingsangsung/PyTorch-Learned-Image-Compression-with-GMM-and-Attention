#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
from .GDN import GDN
from .attention import Attention
from .mambaconv import MambaConv2d
# class Analysis_transform(nn.Module):
#     def __init__(self, num_filters=128):
#         super(Analysis_transform, self).__init__()
#         self.conv_shortcut0 = nn.Conv2d(3, num_filters, 1, stride=2, padding=0)
#         self.conv0 = nn.Conv2d(3, num_filters, 3, stride=2, padding=1)
#         self.conv1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
#         self.leaky_relu1 = nn.LeakyReLU()
#         self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
#         self.leaky_relu2 = nn.LeakyReLU()
#         self.conv_shortcut = nn.Conv2d(num_filters, num_filters, 1, stride=2, padding=0)
#         self.conv3 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
#         self.leaky_relu3 = nn.LeakyReLU()
#         self.conv4 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
#         self.gdn = GDN(num_filters)
#         # self.leaky_relu4 = nn.LeakyReLU()
#         self.conv5 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1, bias=False)
#         self.attention1 = Attention(num_filters)
#         self.attention2 = Attention(num_filters)
#
#
#     def forward(self, x):
#         for i in range(4):
#             if i > 0:
#                 x2 = self.conv1(x)
#                 x2 = self.leaky_relu1(x2)
#                 # print("a 3x3 1")
#                 # print("%d"%(i), x2.shape)
#                 x2 = self.conv2(x2)
#                 x2 = self.leaky_relu2(x2)
#                 # print("b 3x3 1")
#                 # print("%d"%(i), x2.shape)
#                 x = x + x2
#                 # print("resblock result: ", x.shape)
#
#
#             if i == 0:
#                 shortcut_tensor = self.conv_shortcut0(x)
#                 x = self.conv0(x)
#                 x = self.leaky_relu3(x)
#                 # print("c 3x3 2")
#                 # print("%d"%(i), x.shape)
#                 x = self.conv4(x)
#                 # x = self.leaky_relu4(x)
#                 x = self.gdn(x)
#                 # print("d 3x3 1")
#                 # print("%d"%(i), x.shape)
#                 x = x + shortcut_tensor
#                 # print("resblock result: ", x.shape)
#             elif i < 3:
#                 shortcut_tensor = self.conv_shortcut(x)
#                 x = self.conv3(x)
#                 x = self.leaky_relu3(x)
#                 # print("c 3x3 2")
#                 # print("%d"%(i), x.shape)
#                 x = self.conv4(x)
#                 # x = self.leaky_relu4(x)
#                 x = self.gdn(x)
#                 # print("d 3x3 1")
#                 # print("%d"%(i), x.shape)
#                 x = x + shortcut_tensor
#                 # print("resblock result: ", x.shape)
#                 if i == 1:
#                     # Attenation
#                     x = self.attention1(x)
#
#             else:
#                 x = self.conv5(x)
#                 x = self.attention2(x)
#
#         return x

class Analysis_transform(nn.Module):
    def __init__(self, num_filters=128, use_ssm=False):
        super(Analysis_transform, self).__init__()
        # i = 0
        self.b0_shortcut = nn.Conv2d(3, num_filters, 1, stride=2)
        self.b0_layer2 = nn.Conv2d(3, num_filters, 3, stride=2, padding=1)
        self.b0_layer2_relu = nn.LeakyReLU()
        self.b0_layer3 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b0_layer3_GDN = GDN(num_filters)

        # i = 1
        self.b1_layer0 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b1_layer0_relu = nn.LeakyReLU()
        self.b1_layer1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b1_layer1_relu = nn.LeakyReLU()
        self.b1_shortcut = nn.Conv2d(num_filters, num_filters, 1, stride=2)
        self.b1_layer2 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
        self.b1_layer2_relu = nn.LeakyReLU()
        self.b1_layer3 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b1_layer3_GDN = GDN(num_filters)
        self.attention1 = Attention(num_filters)

        # i = 2
        self.b2_layer0 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer0_relu = nn.LeakyReLU()
        self.b2_layer1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer1_relu = nn.LeakyReLU()
        self.b2_shortcut = nn.Conv2d(num_filters, num_filters, 1, stride=2)
        self.b2_layer2 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
        self.b2_layer2_relu = nn.LeakyReLU()
        self.b2_layer3 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer3_GDN = GDN(num_filters)

        # i = 3
        self.b3_layer0 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b3_layer0_relu = nn.LeakyReLU()
        self.b3_layer1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b3_layer1_relu = nn.LeakyReLU()
        self.b3_layer2 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1, bias=False)
        self.attention2 = Attention(num_filters)
        self.downscaler = nn.ModuleList([
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.AvgPool2d(3, stride=2, padding=1),
        ])
        if use_ssm:
            self.mambaconvs = nn.ModuleList([
                 MambaConv2d(3, num_filters, kernel_size=4, stride=4, padding=0, bias=False, dim_preserve=True),
                 MambaConv2d(3, num_filters, kernel_size=4, stride=4, padding=0, bias=False, dim_preserve=True),
                 MambaConv2d(3, num_filters, kernel_size=2, stride=2, padding=0, bias=False, dim_preserve=True),
                 MambaConv2d(3, num_filters, kernel_size=2, stride=2, padding=0, bias=False, dim_preserve=True),
            ])
            self.initial_mambaconv = MambaConv2d(3, 3, kernel_size=4, stride=4, padding=0, bias=False, dim_preserve=True)
        self.use_ssm = use_ssm
        self.num_filters = num_filters

    def forward(self, x):
        interm_signals = []
        x_ = x.clone()
        if self.use_ssm:
             for downscaler, mambaconv in zip(self.downscaler, self.mambaconvs):
                 x_ = downscaler(x_)
                 m = mambaconv(x_)
                 if x_.size(2)%2 == 1 or x_.size(3)%2 == 1:
                      padding = (0, x_.size(3)%2, 0, x_.size(2)%2)
                      m = F.pad(m, padding, "replicate")
                 interm_signals.append(m)
             x = self.initial_mambaconv(x)
        else:
             for downscaler in self.downscaler:
                x_ = downscaler(x_)
                h, w = x_.shape[-2:]
                interm_signals.append(torch.zeros(x_.size(0), self.num_filters, h, w, \
                                                  device=x.device, requires_grad=False))    

        # i = 0
        shortcut0 = self.b0_shortcut(x)
        b0 = self.b0_layer2(x)
        b0 = self.b0_layer2_relu(b0)
        b0 = self.b0_layer3(b0)
        b0 = self.b0_layer3_GDN(b0 + interm_signals[0])
        b0 += shortcut0

        # i = 1
        b1 = self.b1_layer0(b0)
        b1 = self.b1_layer0_relu(b1)
        b1 = self.b1_layer1(b1)
        b1 = self.b1_layer1_relu(b1)
        b1 += b0
        shortcut1 = self.b1_shortcut(b1)
        b1 = self.b1_layer2(b1)
        b1 = self.b1_layer2_relu(b1)
        b1 = self.b1_layer3(b1)
        b1 = self.b1_layer3_GDN(b1 + interm_signals[1])
        b1 += shortcut1
        b1 = self.attention1(b1)

        # i = 2
        b2 = self.b2_layer0(b1)
        b2 = self.b2_layer0_relu(b2)
        b2 = self.b2_layer1(b2)
        b2 = self.b2_layer1_relu(b2)
        b2 += b1
        shortcut2 = self.b2_shortcut(b2)
        b2 = self.b2_layer2(b2)
        b2 = self.b2_layer2_relu(b2)
        b2 = self.b2_layer3(b2)
        b2 = self.b2_layer3_GDN(b2 + interm_signals[2])
        b2 += shortcut2

        # i = 3
        b3 = self.b3_layer0(b2)
        b3 = self.b3_layer0_relu(b3)
        b3 = self.b3_layer1(b3)
        b3 = self.b3_layer1_relu(b3)
        b3 += b2
        b3 = self.b3_layer2(b3) + interm_signals[3]
        b3 = self.attention2(b3)

        return b3

if __name__ == "__main__":
    analysis_transform = Analysis_transform()
    input_image = torch.zeros([1,3,256,256])
    feature = analysis_transform(input_image)
    print(feature.shape)
