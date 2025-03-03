import cv2
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('github.jpg')

# 分离通道
b, g, r = cv2.split(img)

# 显示原始图像及其三个通道
plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 原始图像需要转换颜色空间
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(r, cmap='gray')  # 红色通道
plt.title('Red Channel')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(g, cmap='gray')  # 绿色通道
plt.title('Green Channel')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(b, cmap='gray')  # 蓝色通道
plt.title('Blue Channel')
plt.axis('off')

plt.show()