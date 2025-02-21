import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#matplotlib.pyplot用来画图，首先创建窗口,plt.figure()，然后添加子图位置以及投影方式
# 创建一个新的图形
fig = plt.figure()

# 添加一个三维子图
ax = fig.add_subplot(111, projection='3d')

# 定义平面方程参数 (Ax + By + Cz = D)
A, B, C, D = 1, 2, -1, 0  # 平面方程: x + 2y - z = 0

# 定义网格范围
x_min, x_max = -5, 5
y_min, y_max = -5, 5

# 创建网格数据
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)

# 根据平面方程计算 Z 值
Z = (D - A*X - B*Y) / C

# 绘制平面
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, cmap='viridis')

# 设置轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 设置标题
ax.set_title('3D Plane in a 3D Coordinate System')

# 显示图形
plt.show()