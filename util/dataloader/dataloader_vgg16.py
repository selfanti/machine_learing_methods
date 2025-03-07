from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np
import gzip
from torch.utils.data import DataLoader


def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        # 跳过前16字节的元数据（魔数、图像数量、行数、列数）
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        # 转换为 (样本数, 28, 28) 形状，并添加通道维度（灰度图）
        return data.copy().reshape(-1, 28, 28, 1)  #注意copy，转换成可修改的tensor

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        # 跳过前8字节的元数据
        data=np.frombuffer(f.read(), np.uint8, offset=8)
        return data.copy()  #注意copy，转换成可修改的tensor

def get_img_labels_dataset(type):
    # 加载数据
    if type=='train':
        x_train = load_mnist_images('data/VGG16/train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('data/VGG16/train-labels-idx1-ubyte.gz')
        # 定义转换：归一化到[0,1]并标准化（均值和标准差根据MNIST预设）
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为Tensor并自动归一化到[0,1]
            transforms.Normalize((0.1307,), (0.3081,))  # 网页3、网页6、网页8中的常用参数
        ])
        #应用转换
        x_train = torch.stack([transform(img) for img in x_train])
        y_train = torch.from_numpy(y_train).long()
        # 实例化数据集
        train_dataset = MNISTDataset(x_train, y_train)

        return train_dataset
    elif type=='test':
        x_test = load_mnist_images('data/VGG16/t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels('data/VGG16/t10k-labels-idx1-ubyte.gz')
        # 定义转换：归一化到[0,1]并标准化（均值和标准差根据MNIST预设）
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为Tensor并自动归一化到[0,1]
            transforms.Normalize((0.1307,), (0.3081,))  # 网页3、网页6、网页8中的常用参数
        ])

        #应用转换
        x_test = torch.stack([transform(img) for img in x_test])
        y_test = torch.from_numpy(y_test).long()
        test_dataset = MNISTDataset(x_test, y_test)
        return test_dataset
    else:
        RuntimeError('type error!')
class MNISTDataset(Dataset):
    def __init__(self, images, labels):

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def dataload_vgg16(batch_size,if_check=True):
    train_dataset=get_img_labels_dataset('train')
    test_dataset = get_img_labels_dataset('test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 网页1、网页4推荐参数
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if if_check:
        for images, labels in train_loader:
            print(f"图像张量形状: {images.shape}")  # 应为 [batch_size, 1, 28, 28]
            print(f"标签张量形状: {labels.shape}")  # 应为 [batch_size]
            break
    return train_loader,test_loader
