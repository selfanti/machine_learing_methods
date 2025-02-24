import  numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
# 设置显示所有行
pd.options.display.max_rows = None
# 设置显示所有列
pd.options.display.max_columns = None

def distance_fc(a,b,p):
    if len(a) !=len(b):
        raise  ValueError('length of a and b must be equal')
    else:
        return np.sum(np.abs(a-b)**p)**(1/p)


def encode_dataset_y(dataset):
    income_mapping = {'<=50K': 0, '>50K': 1}
    dataset = dataset.map(income_mapping)
    return dataset
def encode_dataset_x(dataset):
    #先处理多类型的特征
    df_one_hot = pd.get_dummies(dataset,columns=['workclass','education','marital-status','occupation',
                                                 'relationship','race','native-country'])
    #再处理0-1型特征
    gender_mapping = {'Male': 0, 'Female': 1}
    df_one_hot['gender'] = df_one_hot['gender'].map(gender_mapping)
    # gender_mapping = {'<=50K': 0, '>50K': 1}
    # df_one_hot['income'] = df_one_hot['income'].map(gender_mapping)
    #最后处理连续性变量，对于连续性变量，假设各个特征同等重要，因此全部进行归一化
    columns_to_normalize = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    df_normalized = df_one_hot.copy()
    df_normalized[columns_to_normalize] = df_normalized[columns_to_normalize].apply(min_max_scaling)
    return df_normalized
def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())
# Download latest version
path = kagglehub.dataset_download("wenruliu/adult-income-dataset")

print("Path to dataset files:", path)
dataset=pd.read_csv(path+r'\adult.csv')

data_clean = dataset.replace(regex=[r'\?|\.|\$'],value=np.nan)
print(data_clean.isnull().any())

adult = data_clean.dropna(how='any')
print(adult.shape)
adult = adult.drop(['fnlwgt'],axis=1)
print(adult.info())



# 数据分离
col_names = ["age", "workclass", "education", "educational-num", "marital-status", "occupation",
             "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week", "native-country",
             "income"]
X_train, X_test, Y_train, Y_test = train_test_split(adult[col_names[0:13]], adult[col_names[13]], test_size=0.25,
                                          random_state=33)
print("after split\n")
# age               连续型
# workclass         多类别型
# education         多类别型
# educational-num   连续型
# marital-status    多类别型
# occupation        多类别型
# relationship      多类别型
# race              多类别型
# gender            0-1型
# capital-gain      连续型
# capital-loss      连续型
# hours-per-week    连续型
# native-country    多类别型

# income            0-1型


df_encoded_X_train = encode_dataset_x(X_train)
df_encoded_X_test = encode_dataset_x(X_test)
print(df_encoded_X_train.info())
print(df_encoded_X_train.shape)
df_encoded_Y_train=encode_dataset_y(Y_train)
df_encoded_Y_test=encode_dataset_y(Y_test)
print(df_encoded_Y_train.info())
print(df_encoded_Y_train.head())
#所有数据整理完毕
kdt = KDTree(df_encoded_X_train)

# 现在假设我们有一个测试数据点，想要找到它的最近邻居
test_point = np.random.randn(1,101)
print(test_point)
# 查询最近的邻居数量 k
k = 10
dist, ind = kdt.query(test_point, k=k)

# 输出最近邻居的索引和距离
print("Indices of the nearest neighbors:")
print(ind)
print("Distances to the nearest neighbors:")
print(dist)

# 根据最近邻居的索引来确定测试点的类别
nearest_labels = df_encoded_Y_train.to_numpy()
nearest_labels =nearest_labels[ind][0]
most_common_label = np.bincount(nearest_labels).argmax()

print(f"The predicted class for the test point is: {most_common_label}")