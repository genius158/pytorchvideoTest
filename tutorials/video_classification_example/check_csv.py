import pandas as pd
import os

data_path = "video"  # 你的数据路径

# 检查训练数据
train_csv = os.path.join(data_path, "train.csv")
if os.path.exists(train_csv):
    df = pd.read_csv(train_csv)
    print(f"训练样本数：{len(df)}")
    print(f"唯一标签数：{df['label'].nunique()}")
    print(f"标签分布:\n{df['label'].value_counts()}")
    print(f"标签列名：{df.columns.tolist()}")
else:
    print(f"文件不存在：{train_csv}")