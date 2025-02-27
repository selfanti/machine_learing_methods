import torch

# 创建一个 FloatStorage 对象
float_storage = torch.FloatStorage(5)  # 大小为 5 的浮点数存储对象
print(float_storage)

# 初始化存储对象中的数据
float_storage[:] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
print(float_storage)
print(torch.__version__)
print(torch.get_default_device())
a = torch.randn(1, 2, 3, 4, 5)
torch.numel(a)
b=a
b = torch.zeros(4,4)
print(torch.numel(b))