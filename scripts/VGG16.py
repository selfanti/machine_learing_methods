from model.VGG16 import vgg16
import torch
import torch.nn as nn
import numpy as np
from util.dataloader.dataloader_vgg16 import dataload_vgg16

if __name__ =='__main__':
    batch_size=16
    train_epochs=100
    model=vgg16.VGG16()
    train_loader,test_loader=dataload_vgg16(batch_size)
    loss_sum=0
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(train_epochs):
        for batch_images, batch_labels in train_loader:
            # 将数据送入GPU（如有）
            batch_images = batch_images
            batch_labels = batch_labels

            # 模型前向传播与训练逻辑
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: ',epoch,'loss: ',loss_sum/batch_size)
        loss_sum=0
