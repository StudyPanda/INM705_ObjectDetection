import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.models as models

from utils import get_device


class conv_block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super(conv_block, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.batch_norm = nn.BatchNorm2d(out_channels)
    self.leaky_relu = nn.LeakyReLU(0.1)
  
  def forward(self, x):
    x = self.conv(x)
    x = self.batch_norm(x)
    x = self.leaky_relu(x)
    return x

class Yolov1(nn.Module):
  def __init__(self, num_classes = 20, pretrained = False):
    super(Yolov1, self).__init__()
    self.num_boxes = 2
    self.num_classes = num_classes
    if pretrained:
      googlenet = models.googlenet(pretrained=True)
      self.feature_extractor = nn.Sequential(*list(googlenet.children())[:-3])
    else:
      self.feature_extractor = nn.Sequential(conv_block(3, 64, 7, 2, 3), nn.MaxPool2d(2,2),
                                             conv_block(64, 192, 3, 1, 1), nn.MaxPool2d(2,2),
                                             conv_block(192, 128, 1, 1, 0), conv_block(128, 256, 3, 1, 1), conv_block(256, 256, 1, 1, 0), conv_block(256, 512, 3, 1, 1), nn.MaxPool2d(2,2),
                                             conv_block(512, 256, 1, 1, 0), conv_block(256, 512, 3, 1, 1),
                                             conv_block(512, 256, 1, 1, 0), conv_block(256, 512, 3, 1, 1),
                                             conv_block(512, 256, 1, 1, 0), conv_block(256, 512, 3, 1, 1),
                                             conv_block(512, 256, 1, 1, 0), conv_block(256, 512, 3, 1, 1),
                                             conv_block(512, 512, 1, 1, 0), conv_block(512, 1024, 3, 1, 1), nn.MaxPool2d(2,2),
                                             conv_block(1024, 512, 1, 1, 0), conv_block(512, 1024, 3, 1, 1),
                                             conv_block(1024, 512, 1, 1, 0), conv_block(512, 1024, 3, 1, 1),
                                              )
    #feature extractor backbone
    self.conv_block_final = nn.Sequential(conv_block(1024, 1024, 3, 1, 1), conv_block(1024, 1024, 3, 2, 1))

    #object detector head
    self.fc1 = nn.Linear(7*7*1024, 4096)
    self.batch_norm = nn.BatchNorm1d(4096)
    self.fc2 = nn.Linear(4096, 7*7*(5*self.num_boxes + self.num_classes)) 
    
    #activation function
    self.leaky_relu = nn.LeakyReLU(0.1)
  
  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.conv_block_final(x)
    x = x.flatten(start_dim = 1)
    x = self.fc1(x)
    x = self.batch_norm(x)
    x = self.leaky_relu(x)
    x = self.fc2(x)
    x = x.view(-1, 7, 7, self.num_boxes *5 + self.num_classes) #bounding box output is [confidence, x, y, w, h]

    x = torch.sigmoid(x)
    return x

    
  


    