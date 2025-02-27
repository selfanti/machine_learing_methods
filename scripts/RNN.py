from model.RNN import rnn
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
#定义用于回归的RNN模型
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
    print("组内排序前:\n")
    print(train_df.head(10))
    print(test_df.head(10))
    train_df_sorted=train_df.groupby("saledate",group_keys=False).apply(lambda group:group.sort_values("bedrooms").reset_index(drop=True))
    test_df_sorted = test_df.groupby("saledate", group_keys=False).apply(
        lambda group: group.sort_values("bedrooms").reset_index(drop=True))
    train_df_sorted=train_df_sorted[['MA','bedrooms']]
    test_df_sorted = test_df_sorted[['MA', 'bedrooms']]
    print("组内排序后:\n")
    print(train_df_sorted.head(10))
    print(test_df_sorted.head(10))
    columns_to_normalize = ['MA', 'bedrooms']
    df_normalized_train = train_df_sorted.copy()
    df_normalized_train[columns_to_normalize] = df_normalized_train[columns_to_normalize].apply(min_max_scaling)
    df_normalized_test = test_df_sorted.copy()
    df_normalized_test[columns_to_normalize] = df_normalized_test[columns_to_normalize].apply(min_max_scaling)
    array_train=df_normalized_train.values.reshape(-1,4,2)
    array_test = df_normalized_train.values.reshape(-1,4,2)
    # 查看结果
    # print('处理后的数据示例')
    # print('训练集：\n')
    # print(array_train)
    # print('测试集：\n')
    # print(array_test )
    return array_train,array_test

def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())


if __name__ =="__main__":
    #house和unit分开训练
    house_type='house'
    path= r"data/RNN/ma_lga_12345.csv"
    train_set,test_set=data=load_data(path,house_type)
    print(train_set)



