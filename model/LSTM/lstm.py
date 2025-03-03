import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model.RNN import rnn

class ManualLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        #初始化权重矩阵
        # 初始化权重矩阵（输入门、遗忘门、输出门、候选细胞）
        self.W_ii = torch.nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_hi = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = torch.nn.Parameter(torch.Tensor(hidden_dim))

        self.W_if = torch.nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_hf = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = torch.nn.Parameter(torch.Tensor(hidden_dim))

        self.W_io = torch.nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_ho = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = torch.nn.Parameter(torch.Tensor(hidden_dim))

        self.W_ig = torch.nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_hg = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = torch.nn.Parameter(torch.Tensor(hidden_dim))

        # Xavier初始化
        torch.nn.init.xavier_uniform_(self.W_ii)
        torch.nn.init.xavier_uniform_(self.W_hi)
        torch.nn.init.constant_(self.b_i, 0)

        torch.nn.init.xavier_uniform_(self.W_if)
        torch.nn.init.xavier_uniform_(self.W_hf)
        torch.nn.init.constant_(self.b_f, 0)

        torch.nn.init.xavier_uniform_(self.W_io)
        torch.nn.init.xavier_uniform_(self.W_ho)
        torch.nn.init.constant_(self.b_o, 0)

        torch.nn.init.xavier_uniform_(self.W_ig)
        torch.nn.init.xavier_uniform_(self.W_hg)
        torch.nn.init.constant_(self.b_g, 0)

    def forward(self,x,intial_states=None):

        batch_size,seq_len, _=x.shape
        hidden_seq=[]

        if intial_states==None:
            h_t=torch.zeros(batch_size,self.hidden_dim).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        else:
            h_t,c_t=intial_states

        for t in range(seq_len):
            x_t=x[:,t,:]

            #门控信号
            i_t=torch.sigmoid(x_t@self.W_ii.T+h_t@self.W_hi.T+self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_if.T + h_t @ self.W_hf.T + self.b_f)
            o_t = torch.sigmoid(x_t @ self.W_io.T + h_t @ self.W_ho.T + self.b_o)
            g_t = torch.tanh(x_t @ self.W_ig.T + h_t @ self.W_hg.T + self.b_g)
            #更新细胞状态和隐藏状态
            c_t=f_t*c_t+i_t*g_t
            h_t=o_t*torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq=torch.cat(hidden_seq,dim=0)
        return hidden_seq.transpose(0,1),(h_t,c_t)




