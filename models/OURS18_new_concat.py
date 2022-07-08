##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Feng Li
## School of Computer Science & Engineering, South China University of Technology
## Email: csfengli@mail.scut.edu.cn
## Copyright (c) 2019
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

import encoding
import torchvision.models as resnet
from FAPool.FAP import FAP

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:05:26 2018
Generate histogram layer
@author: jpeeples
"""

import torch
import torch.nn as nn
import numpy as np

from Demo_Parameters import Network_parameters
# Name of dataset
Dataset = Network_parameters['Dataset']

# Model(s) to be used
model_name = Network_parameters['Model_names'][Dataset]

# Number of classes in dataset
num_classes = Network_parameters['num_classes'][Dataset]

# Number of runs and/or splits for dataset
numRuns = Network_parameters['Splits'][Dataset]

# Number of bins and input convolution feature maps after channel-wise pooling
numBins = Network_parameters['numBins']
num_feature_maps = Network_parameters['out_channels'][model_name]

# Local area of feature map after histogram layer
feat_map_size = Network_parameters['feat_map_size']


class HistogramLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dim=2, num_bins=4,
                 stride=1, padding=0, normalize_count=True, normalize_bins=True,
                 count_include_pad=False,
                 ceil_mode=False):

        # inherit nn.module
        super(HistogramLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode

        # For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if self.dim == 1:
            self.bin_centers_conv = nn.Conv1d(self.in_channels, self.numBins * self.in_channels, 1,
                                              groups=self.in_channels, bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv1d(self.numBins * self.in_channels,
                                             self.numBins * self.in_channels, 1,
                                             groups=self.numBins * self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool1d(self.filt_dim, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        # Image Data
        elif self.dim == 2:
            self.bin_centers_conv = nn.Conv2d(self.in_channels, self.numBins * self.in_channels, 1,
                                              groups=self.in_channels, bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv2d(self.numBins * self.in_channels,
                                             self.numBins * self.in_channels, 1,
                                             groups=self.numBins * self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool2d(self.kernel_size, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            self.bin_centers_conv = nn.Conv3d(self.in_channels, self.numBins * self.in_channels, 1,
                                              groups=self.in_channels, bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv3d(self.numBins * self.in_channels,
                                             self.numBins * self.in_channels, 1,
                                             groups=self.numBins * self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool3d(self.filt_dim, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        else:
            raise RuntimeError('Invalid dimension for histogram layer')

    def forward(self, xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window

        # Pass through first convolution to learn bin centers
        xx = self.bin_centers_conv(xx)

        # Pass through second convolution to learn bin widths
        xx = self.bin_widths_conv(xx)

        # Pass through radial basis function
        xx = torch.exp(-(xx ** 2))

        # Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if (self.normalize_bins):
            xx = self.constrain_bins(xx)

        # Get localized histogram output, if normalize, average count
        if (self.normalize_count):
            xx = self.hist_pool(xx)
        else:
            xx = np.prod(np.asarray(self.hist_pool.kernel_size)) * self.hist_pool(xx)

        return xx

    def constrain_bins(self, xx):
        # Enforce sum to one constraint across bins
        # Time series/ signal Data
        if self.dim == 1:
            n, c, l = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, l).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

            # Image Data
        elif self.dim == 2:
            n, c, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

            # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            n, c, d, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, d, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

        else:
            raise RuntimeError('Invalid dimension for histogram layer')

        return xx

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self,device, sigma=0.2, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        nclass = opt.n_classes
        # copying modules from pretrained models
        if opt.backbone.lower() == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.use_pretrained)
        elif opt.backbone.lower() == 'resnet50':
            self.backbone = models.resnet50(pretrained=opt.use_pretrained)
        elif opt.backbone.lower() == 'resnet101':
            self.backbone = models.resnet101(pretrained=opt.use_pretrained)
        elif opt.backbone.lower() == 'resnet152':
            self.backbone = models.resnet152(pretrained=opt.use_pretrained)  # , dilated=False
        else:
            raise Exception('unknown backbone: {}'.format(opt.backbone.lower()))
        self.dim = opt.dim
        
        self.histogram_layer = HistogramLayer(int(num_feature_maps/(feat_map_size*numBins)),
                                  Network_parameters['kernel_size'][model_name],
                                  num_bins=numBins,stride=Network_parameters['stride'],
                                  normalize_count=Network_parameters['normalize_count'],
                                  normalize_bins=Network_parameters['normalize_bins'])
        # print("num_feature_maps:", num_feature_maps)
        # print("feat_map_size:", feat_map_size)
        # print("in channels:", int(num_feature_maps/(feat_map_size*numBins)))
        n_codes = 8
        encoding_out = 64
        self.head = nn.Sequential(
            # nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=512, K=n_codes),
            encoding.nn.View(-1, 512 * n_codes),
            encoding.nn.Normalize(),
            nn.Linear(512 * n_codes, encoding_out),
            nn.BatchNorm1d(encoding_out),
        )

        self.conv_before_mfs = nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU())
        self.mfs = FAP(opt, D=1, K=self.dim)

        hist_out = 64
        fractal_out = 48

        self.fc = nn.Sequential(
            encoding.nn.Normalize(),
            #nn.ELU(),
            #nn.Dropout(0.20),
            #nn.AlphaDropout(p=0.1),
            nn.Linear(hist_out+encoding_out+fractal_out, 128),
            encoding.nn.Normalize(),
            #nn.ELU(),
            #nn.Dropout(0.50),
            nn.Linear(128, nclass)  # nclass
        )
        
        self.hist_fc = nn.Sequential(
            encoding.nn.Normalize(),
            nn.Linear(512, hist_out))

        self.UP = nn.ConvTranspose2d(512, 512, 3, 2, groups=512)  # group operation is important for compact model size
        self.conv_down = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.noise = GaussianNoise(torch.device("cuda"))

    def forward(self, x):
        #print("in ours18`")
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            var_input = x
            while not isinstance(var_input, Variable):
                var_input = var_input[0]
            _, _, h, w = var_input.size()
        else:
            raise RuntimeError('unknown input type: ', type(x))
        # print("original")
        # print("input shape:", x.shape)
        x = self.backbone.conv1(x)
        # x = self.dep(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # up = nn.Upsample(size=(14,14)) #mode='bilinear',align_corners=True
        #x = self.noise(x)
        up = self.UP(x)
        # import pdb
        # pdb.set_trace()
        # print(up.shape)[32, 512, 15, 15]
        c = self.conv_before_mfs(up)
        c0 = c[:, 0, :, :].unsqueeze_(1)
        c1 = c[:, 1, :, :].unsqueeze_(1)
        c2 = c[:, 2, :, :].unsqueeze_(1)

        fracdim0 = self.mfs(c0).squeeze_(-1).squeeze_(-1)
        fracdim1 = self.mfs(c1).squeeze_(-1).squeeze_(-1)
        fracdim2 = self.mfs(c2).squeeze_(-1).squeeze_(-1)
        x0 = self.head(x)
        x2 = torch.cat((fracdim0, fracdim1, fracdim2), 1)
        x_temp = self.conv_down(x)
        x_hist = torch.flatten(self.histogram_layer(x_temp), start_dim=1)
        x_hist = self.hist_fc(x_hist)
        
        #x = torch.cat((x1, x2, x_hist), 1)
        x = torch.cat((x0,x2, x_hist), 1)
        #x = self.noise(x)
        #print("x shape is:", x.shape)

        x = self.fc(x)

        return x


def test():
    net = Net(nclass=23).cuda()
    print(net)

    test = net.cpu().state_dict()
    print('=============================================================================')
    for key, v in test.items():
        print(key)
    net.cuda()
    x = Variable(torch.randn(1, 3, 224, 224)).cuda()
    y = net(x)
    print(y)
    params = net.parameters()
    sum = 0
    for param in params:
        sum += param.nelement()
    print('Total params:', sum)


if __name__ == "__main__":
    test()
