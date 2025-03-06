# torch如何计算梯度
## 直接计算
torch中的模型常常包含多个层，也就是多种运算，
可以直接展开。使用一个简单例子，即线性层加激活函数。样本为$(x,t)$
$$ z=wx+b$$
$$y=\sigma(z)$$
$$\mathcal{L}=\frac{1}{2}(y-t)^2$$
$$R=\frac{1}{2}w^2$$
$$\mathcal{L_reg}=\mathcal{L}+\lambda R$$
展开后为
$$\begin{aligned}\mathcal{L_reg}&=\frac{1}{2}(y-t)^2+\lambda \frac{1}{2}w^2\\
&=\frac{1}{2}(\sigma(wx+b)-t)^2+\lambda \frac{1}{2}w^2\\
\end{aligned}$$
其中$\lambda$是超参数  
为了进行梯度下降，需要对线性层的参数$w$和$b$进行梯度下降，于是有
$$
\begin{aligned}
\frac{\partial{\mathcal{L}}}{\partial{w}}=\\
\frac{\partial{\mathcal{L}}}{\partial{b}}=\\
\end{aligned}
$$