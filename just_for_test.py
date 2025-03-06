import torch.nn as nn
import torch
# 创建嵌入层（词典大小20，向量维度5）
embedding = nn.Embedding(20, 5)
# 输入索引张量
input = torch.LongTensor([0, 1, 2])  # 3个单词的索引
output = embedding(input)  # 输出形状：(3, 5)
print(output)

#每次编码方式都不一样