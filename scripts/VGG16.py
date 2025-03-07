from model.VGG16 import vgg16
import torch
import torch.nn as nn
import numpy as np
from util.dataloader.dataloader_vgg16 import dataload_vgg16

if __name__ =='__main__':
    model=vgg16.VGG16()
    train_loader,test_loader=dataload_vgg16(16)
