from model.LSTM import lstm
import torch
import torch.nn as nn
import numpy as np
class Regression_LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.LSTM=lstm.ManualLSTM(input_dim,hidden_dim)
        self.fc=nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        batch_size,seq_len,_=x.shape

        lstm_output=self.LSTM(x)
        hidden_seq, _ = self.LSTM(x)  # 取第一个返回值（隐藏序列）
        output = self.fc(hidden_seq[:, -1, :])  # 取最后一个时间步
        return output
if __name__=='__main__':
    # 测试代码
    input_dim = 5
    hidden_dim = 10
    output_dim = 1
    model = Regression_LSTM(input_dim, hidden_dim, output_dim)
    x = torch.randn(3, 7, input_dim)  # (batch=3, seq_len=7, input_dim=5)
    y = model(x)
    print(y.shape)  # 预期输出: torch.Size([3, 1])
    criterion = nn.MSELoss()
    y_true = torch.randn(3, 1)
    loss = criterion(y, y_true)
    print(loss)
    loss.backward()
