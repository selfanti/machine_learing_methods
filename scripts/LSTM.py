from model.LSTM import lstm
from model.RNN import rnn
import torch
import torch.nn as nn
from RNN import  load_data,create_dataset,draw_loss_epochs
import numpy as np

from scripts.RNN import train_epochs


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
    input_dim = 2
    hidden_dim = 64
    output_dim = 2
    train_epochs=100
    model = Regression_LSTM(input_dim, hidden_dim, output_dim)
    house_type='house'
    path= r"U:\Users\Enlink\PycharmProjects\machine_learning_user\data\RNN\ma_lga_12345.csv"
    train_set,test_set=data=load_data(path,house_type)
    print('train_set.shape',train_set.shape)
    print('test_set.shape',test_set.shape)
    X_list, Y_list = train_list = create_dataset(train_set, 6)
    print('X_list: ',len(X_list),'Y_list: ',len(Y_list))
    criterion = nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    loss_list=[]
    loss_mean_list=[]
    print("Start Training...")
    for epoch in range(train_epochs):
        for i in range(40):
            optimizer.zero_grad()
            output=model(X_list[i])
            loss=criterion(output,Y_list[i])
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()

        loss_mean_list.append(sum(loss_list) / len(loss_list))
        loss_list = []
        print(f"epoch:{epoch}  loss:{loss_mean_list[-1]}")
    torch.save(model.state_dict(),r"U:\Users\Enlink\PycharmProjects\machine_learning_user\weights_model\lstm_weights.pth")
    model.eval()
    predict_value=[]
    eval_loss=[]
    copy_train_data=train_set.clone()
    print('copy_train_data:',copy_train_data.shape)

    for j in range(40,43):
        this_window=copy_train_data[:,j:j+6,:]
        output=model(this_window)
        copy_train_data=torch.concat([copy_train_data,output.unsqueeze(1)],dim=1)
        predict_value.append(output)
        loss=criterion(output,test_set[:,j-40,:])
        eval_loss.append(loss)

    draw_loss_epochs(loss_mean_list, train_epochs)
    print('copy_train_data.shape: ',copy_train_data.shape)
    torch.concat([train_set,test_set],dim=1)
