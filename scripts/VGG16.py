from model.VGG16 import vgg16
import torch
import torch.nn as nn
import numpy as np
from util.dataloader import dataloader_vgg16

if __name__ =='__main__':
    model=vgg16.VGG16
    dataset=dataloader_vgg16.CustomDataset()
    dataset.dataloader(batch_size=8)




