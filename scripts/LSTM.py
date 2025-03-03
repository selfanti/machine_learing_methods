import model.LSTM as LSTM
import torch
import torch.nn as nn
import numpy as np
class Regression_LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.LSTM=LSTM.ManualLSTM(input_dim,hidden_dim)
