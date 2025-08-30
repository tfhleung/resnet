#%%
import torch
import torch.nn as nn
from torchsummary import summary

from collections import OrderedDict
#%%
# Base class for the resnet blocks.  Generally, the resnet is constructed by a series of blocks with a skip connection added to the output which allows
# gradients to propogate more easily during the training process.  When the input and output channels have different dimensions, additional post-processing 
# is required for defining the skip connection so that the dimensions match correctly.  This class abstracts that process and defines a generic interface 
# which can be easily extended to the basic and bottleneck blocks used in the resnet.
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.blocks = nn.Identity()
    self.skip = nn.Identity()
    self.relu = nn.ReLU()

  def forward(self, x):
    if self.in_channels != self.out_channels:
      identity = self.skip(x)
    else:
      identity = x

    x = self.blocks(x)
    x += identity
    return self.relu(x)
  
  def _conv_bn(self, in_channels, out_channels, kernel_size, **kwargs):
    return nn.Sequential( OrderedDict({'conv': nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
                                    'bn': nn.BatchNorm2d(num_features=out_channels)}) )

# Definition of the basic block used in the resnet 18/34, which consists of a series of conv/bn/relu layers.  The basic block consists of two conv blocks in 
# series with kernel sizes of 3x3.
class BasicBlock(ResidualBlock):
  def __init__(self, in_channels, out_channels, activation = nn.ReLU, expansion = 1, downsampling = 1):
    super().__init__(in_channels, out_channels)
    self.expansion = expansion
    self.downsampling = downsampling

    self.blocks = nn.Sequential(
        self._conv_bn(self.in_channels, self.out_channels, kernel_size = 3, stride = self.downsampling, padding = 1, bias = False),
        activation(),
        self._conv_bn(self.out_channels, self.out_channels*self.expansion, kernel_size = 3, padding = 1, bias = False)
        )

    if self.in_channels != self.out_channels*self.expansion:
      self.skip = self._conv_bn(self.in_channels, self.out_channels*self.expansion, kernel_size = 1, stride = self.downsampling, bias = False) 
           
# Definition of the bottleneck block used in the resnet 50/101/152, which consists of a series of conv/bn/relu layers.  To reduce the overall number of parameters in
# the model, the bottleneck design is adpoted where the 1x1 conv layer is used to reduce the number of inputs to the subsequent 3x3 conv layer.  A final
# 1x1 layer is used to restore the number of outputs from the final layer.
class BottleneckBlock(ResidualBlock):
  def __init__(self, in_channels, out_channels, activation = nn.ReLU, expansion = 4, downsampling = 1):
    super().__init__(in_channels, out_channels)
    self.expansion = expansion
    self.downsampling = downsampling

    self.blocks = nn.Sequential(
        self._conv_bn(self.in_channels, self.out_channels, kernel_size = 1, stride = self.downsampling, bias = False),
        activation(),
        self._conv_bn(self.out_channels, self.out_channels, kernel_size = 3, stride = self.downsampling, padding = 1, bias = False),
        activation(),
        self._conv_bn(self.out_channels, self.out_channels*self.expansion, kernel_size = 1, bias = False),
        )

    if self.in_channels != self.out_channels*self.expansion:
      self.skip = self._conv_bn(self.in_channels, self.out_channels*self.expansion, kernel_size = 1, stride = self.downsampling, bias = False) 

class ResNetEncoder(nn.Module):
  def __init__(self, in_channels, block = BasicBlock, block_size = [64, 128, 256, 512], num_layers = [3, 4, 6, 3], activation = nn.ReLU):
    super().__init__()
    self.block = block
    self.block_size = block_size
    self.num_layers = num_layers
    self.activation = activation

    self.stem = nn.Sequential(
       nn.Conv2d(in_channels, block_size[0], kernel_size = 7, stride = 2, padding = 3, bias = False),
       nn.BatchNorm2d(block_size[0]),
       activation(),
       nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    )

    self.list = []
    for i, num in enumerate(num_layers):
      for j in range(num):
          if i > 0 and j == 0:
            self.list.append(block(in_channels = block_size[i-1], out_channels = block_size[i]))
          else:
            self.list.append(block(in_channels = block_size[i], out_channels = block_size[i]))           

    self.layers = nn.Sequential(*self.list)

  def forward(self, x):
      x = self.stem(x)
      x = self.layers(x)
      return x

class ResNetDecoder(nn.Module):
   def __init__(self, in_features, n_classes, **kwargs):
      super().__init__()
      self.pool = nn.AdaptiveAvgPool2d((1,1))
      self.linear = nn.Linear(in_features, n_classes)

   def forward(self, x):
      x = self.pool(x)
      x = torch.flatten(x,1)
      #output needs to flatten vector before input into linear layer
      x = self.linear(x)
      return x

class ResNet(nn.Module):
  def __init__(self, in_channels, n_classes, *args, **kwargs):
    super().__init__()
    self.encoder = ResNetEncoder(in_channels, block = BasicBlock, block_size = [64, 128, 256, 512], num_layers = [3, 4, 5, 3], activation = nn.ReLU)
    self.decoder = ResNetDecoder(self.encoder.block_size[-1], n_classes, **kwargs)
    
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

#%%
if __name__ == "__main__":
  pass