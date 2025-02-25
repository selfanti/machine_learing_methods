import pandas as pd

# 创建示例 DataFrame
data = {
    'Color': ['Red', 'Green', 'Blue', 'Red'],
    'Size': ['S', 'M', 'L', 'XL'],
    'male':['female','male','dont know','?']
}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)

# 使用 get_dummies 进行独热编码
df_encoded = pd.get_dummies(df)

print("\n独热编码后的 DataFrame:")
print(df_encoded)