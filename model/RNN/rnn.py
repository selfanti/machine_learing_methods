import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.W_xh=nn.Parameter(torch.randn(input_dim,hidden_dim)*0.01)
        self.W_hh=nn.Parameter(torch.randn(hidden_dim,hidden_dim)*0.01)
        self.b_h=nn.Parameter(torch.zeros(hidden_dim))
        self.hidden_dim=hidden_dim

    def forward(self,x_seq):
        batch_size,seq_len,_=x_seq.shape
        h_t=torch.zeros(batch_size,self.hidden_dim)
        hidden_states=[]
        for i in range(seq_len):

            x_t=x_seq[:,i,:]
            h_t=torch.tanh(x_t@self.W_xh+h_t@self.W_hh+self.b_h)
            hidden_states.append(h_t.unsqueeze(1))
        return torch.cat(hidden_states,dim=1)