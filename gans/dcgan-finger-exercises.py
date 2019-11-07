# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:03:53 2019

@author: jorda
"""
import torch
import torchvision
import matplotlib.pyplot as pyplot
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

#taken from the pytorch tutorial on GAN faces generation


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

def gen_z(batch_size= 64, feature_length= 100):
  z= np.random.normal(0, 1, (batch_size, feature_length))
  z= z.reshape((batch_size, feature_length, 1, 1))
  z= torch.tensor(z).float().to(device)
  return z

z= gen_z()

class Generator(nn.Module): 
    def __init__(self, feature_length, channels): 
        super(Generator, self).__init__()
        self.features = nn.Sequential(
                nn.ConvTranspose2d(feature_length, 64 * 8, kernel_size = 4, stride = 1, padding= 0, bias = False),
                nn.BatchNorm2d(64 * 8), 
                nn.ReLU(True), 
                
#                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias = False), 
#                nn.BatchNorm2d(64 * 4), 
#                nn.ReLU(True), 
#                
#                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias = False), 
#                nn.BatchNorm2d(64 * 2), 
#                nn.ReLU(True), 
#                
#                nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias = False), 
#                nn.BatchNorm2d(64), 
#                nn.ReLU(True), 
#                
#                nn.ConvTranspose2d(64, channels, 4, 2, 1, bias = False), 
#                nn.Tanh()
        )
        
    def forward(self, z):
        return self.features(z)

gen = Generator(100, 3)
print(gen(z).shape)