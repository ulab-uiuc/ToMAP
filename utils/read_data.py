import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('/data/ph16/TinyZero/datasets/debate_anthropic/test.parquet')

# 获取第一行数据，并转换为字典
for i in range(1):
    row_dict = df.iloc[i].to_dict()
    print(row_dict)
