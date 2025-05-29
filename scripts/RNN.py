from model.RNN import rnn
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
#定义用于回归的RNN模型
train_epochs=50
class RNN_Prediction(nn.Module):
    def __init__(self,inputdim,hiddendim,outputdim):
        super().__init__()
        self.rnn=rnn.MyRNN(inputdim,hiddendim)
        self.fc=nn.Linear(hiddendim,outputdim)
    def forward(self,x):
        rnn_out=self.rnn(x)
        output=self.fc(rnn_out)
        output_final=output[:,-1,:]
        return output_final

def load_data(path,type):

    if type=='house':
        lines_number=4
    else:
        lines_number=0
    # 读取数据并解析日期（注意日期格式是日/月/年）
    df = pd.read_csv(path, parse_dates=["saledate"], dayfirst=True)

    # 1. 过滤掉 unit 类型的数据
    house_df = df[df["type"] == type].copy()

    # 2. 按日期排序（升序排列）
    house_df_sorted = house_df.sort_values(by="saledate")

    # 3. 重置索引（可选步骤，使索引从0开始连续）
    house_df_sorted = house_df_sorted.reset_index(drop=True)
    house_df_sorted_new = house_df_sorted[lines_number:].reset_index(drop=True)  # 删除原索引并重置
    split_date = pd.to_datetime("2018-12-31")
    train_df = house_df_sorted_new[house_df_sorted_new["saledate"] <= split_date]
    test_df = house_df_sorted_new[house_df_sorted_new["saledate"] > split_date]
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df_sorted=train_df.groupby("saledate",group_keys=False).apply(lambda group:group.sort_values("bedrooms").reset_index(drop=True))
    test_df_sorted = test_df.groupby("saledate", group_keys=False).apply(
        lambda group: group.sort_values("bedrooms").reset_index(drop=True))
    train_df_sorted=train_df_sorted[['MA','bedrooms']]
    test_df_sorted = test_df_sorted[['MA', 'bedrooms']]
    columns_to_normalize = ['MA', 'bedrooms']
    df_normalized_train = train_df_sorted.copy()
    df_normalized_train[columns_to_normalize] = df_normalized_train[columns_to_normalize].apply(min_max_scaling)
    df_normalized_test = test_df_sorted.copy()
    df_normalized_test[columns_to_normalize] = df_normalized_test[columns_to_normalize].apply(min_max_scaling)
    array_train_list=[]
    array_test_list =[]
    for i in range(46):
        array_train_list.append(torch.from_numpy(df_normalized_train[i*4:i*4+4].values.astype(np.float32)).unsqueeze(1))
    for j in range(3):
        array_test_list.append(torch.from_numpy(df_normalized_test[j*4:j*4+4].values.astype(np.float32)).unsqueeze(1))
    array_train=torch.cat(array_train_list,dim=1)
    array_test =torch.cat(array_test_list,dim=1)
    return array_train,array_test

def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())
def create_dataset(data, time_steps):
    x, y = [], []
    for i in range(data.shape[1] - time_steps):
        x.append(data[:,i:i+time_steps,:])
        y.append(data[:,i+time_steps,:])

    return x,y
def draw_loss_epochs(loss_mean_list,epochs):
    # 生成示例数据（假设 x 已排序）
    x = np.linspace(0, epochs, epochs)
    # 绘制平滑曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x, loss_mean_list, '-', color='blue', linewidth=2, label='Smooth Curve')
    plt.title("Smooth Curve of Ordered Data")
    plt.xlabel("X")
    plt.ylabel("loss_mean_list")
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ =="__main__":
    #house和unit分开训练
    house_type='house'
    path= r"U:\Users\Enlink\PycharmProjects\machine_learning_user\data\RNN\ma_lga_12345.csv"
    train_set,test_set=data=load_data(path,house_type)
    print('train_set.shape:',train_set.shape)
    print('test_set.shape:',test_set.shape)
    #训练时的窗口大小为6
    X_list,Y_list=train_list=create_dataset(train_set,6)
    print('X_list.shape:',X_list[0].shape)
    print('Y_list.shape:',Y_list[0].shape)
    #X_list = X_list.reshape(X_list.shape[0], X_list.shape[1], 1)  # 转换为3D输入
    model=RNN_Prediction(2,128,2)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    criterion=nn.MSELoss()
    loss_list=[]
    loss_mean_list=[]
    print("训练轮数:",train_epochs)
    for epoch in range(train_epochs):
        for i in range(40):
            optimizer.zero_grad()
            output=model(X_list[i])
            loss=criterion(output,Y_list[i])
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()

        loss_mean_list.append(sum(loss_list)/len(loss_list))
        loss_list=[]
        print(f"epoch:{epoch}  loss:{loss_mean_list[-1]}")
    torch.save(model.state_dict(),
               r"U:\Users\Enlink\PycharmProjects\machine_learning_user\weights_model\rnn_weights.pth")
    draw_loss_epochs(loss_mean_list,train_epochs)





