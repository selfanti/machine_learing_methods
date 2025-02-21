
import numpy as np
import matplotlib.pyplot as plt
# 设置随机种子以便结果可复现
np.random.seed(42)
import math

class perceptron:
    def __init__(self,inputdim,outputdim=1,lr=0.1):
        self.w=np.random.randn(1,inputdim)
        self.b=np.random.uniform(-5,5)
        self.lr=lr

    def forward(self,input):
        output=self.w @ input+self.b
        return output

    def compute_loss(self,output,targets):
        return np.mean((output-targets)**2)

    def train(self,input,target):
        output=self.forward(input)
        grad_w=-1*output*input
        grad_b=-output
        if output*target<0:
            #注意感知机算法只针对错误分类点进行调参
            self.w+=self.lr*grad_w
            self.b+=self.lr*grad_b
    def eval(self,inputs,targets):
        #感知机算法的损失同样参考李航的书中所示
        loss=0
        for index,i in enumerate(inputs):
            loss+=max(0,-1*targets[index]*(self.w @ i+self.b))
        return loss


input=np.array([[0,1],[1,3],[-1,1],[-1,0],[5,0],[4,-1],[-0.5,-0.2],[0,-2]],dtype=np.float32)  #自拟的二维输入点
targets=[1,1,1,1,-1,-1,-1,-1] #分类标签，感知机算法只能用来二分类
model=perceptron(2,1)
for epoch in range(300):
    index=np.random.randint(0,8) #参考李航的深度学习，随机选择输入的点
    model.train(input[index],targets[index])
print('w:',model.w,' b:',model.b)
#打印测试误差，这里图方便使用了训练用的样本
print('loss:',model.eval(input,targets))

fig=plt.figure()
son_pic=fig.add_subplot(111)
#转置后提取x行和y行
input_x=input.transpose()[0]
input_y=input.transpose()[1]
#画出样本点
son_pic.scatter(input_x,input_y)
#增加标题
son_pic.set_title('classfication by perceptron')
line_x=np.linspace(-10,10,100)
line_y=(-model.b-model.w[0][1]*line_x)/model.w[0][0]
son_pic.plot(line_x,line_y)
plt.show()