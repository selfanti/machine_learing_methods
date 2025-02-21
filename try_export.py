import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便结果可复现
np.random.seed(42)


# 定义生成数据的函数
def generate_linearly_separable_data(n_samples_per_class, w, b):
    """
    生成线性可分类的数据。

    参数:
    - n_samples_per_class: 每个类别的样本数量
    - w: 超平面的权重向量 (例如，在二维情况下为 [w1, w2])
    - b: 超平面的偏置项

    返回:
    - X: 样本特征矩阵 (形状为 [n_samples, n_features])
    - y: 样本标签 (形状为 [n_samples])
    """
    # 初始化空列表来存储数据和标签
    X = []
    y = []

    # 随机生成正类 (+1) 的数据点
    for _ in range(n_samples_per_class):
        while True:
            # 在 [-1, 1] 范围内随机生成一个点
            x = np.random.rand(2) * 2 - 1
            # 计算该点与超平面的距离
            distance = np.dot(w, x) + b
            if distance > 0:
                X.append(x)
                y.append(1)
                break

    # 随机生成负类 (-1) 的数据点
    for _ in range(n_samples_per_class):
        while True:
            # 在 [-1, 1] 范围内随机生成一个点
            x = np.random.rand(2) * 2 - 1
            # 计算该点与超平面的距离
            distance = np.dot(w, x) + b
            if distance < 0:
                X.append(x)
                y.append(-1)
                break

    # 将列表转换为 NumPy 数组
    X = np.array(X)
    y = np.array(y)

    return X, y


# 定义超平面参数
w = np.array([1.0, -1.0])  # 权重向量
b = 0.0  # 偏置项

# 生成数据
n_samples_per_class = 100
X, y = generate_linearly_separable_data(n_samples_per_class, w, b)

# 打印前几条数据以检查
print("Sample data points:")
print(X[:5])
print("Corresponding labels:")
print(y[:5])

# 可视化生成的数据
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='-1')

# 绘制分割超平面
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.sign(w[0] * xx + w[1] * yy + b)
plt.contour(xx, yy, Z, levels=[0], colors='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linearly Separable Data')
plt.legend()
plt.show()