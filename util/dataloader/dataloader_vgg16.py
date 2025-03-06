import os
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

class CustomDataset(Dataset):
    def __init__(self,root_dir=None):
        self.root_dir=root_dir
        self.train_image_paths=[]  #图像路径
        self.train_labels=[]       #标签路径
        self.val_image_paths=[]
        self.val_labels=[]
        self.test_image_paths=[]
        self.test_labels=[]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_dataset = datasets.MNIST(root='data/VGG16/train/images', train=True, download=False, transform=self.transform)
        self.test_dataset = datasets.MNIST(root='data/VGG16/test/images', train=False, download=False,
                                      transform=self.transform)  # train=True训练集，=False测试集


    def dataloader(self,batch_size):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader,test_loader


