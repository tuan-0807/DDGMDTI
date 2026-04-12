import pandas as pd

# 读取Excel文件
df = pd.read_csv("./DrugBank_datasets/seed/train.csv")

# 定义函数来修剪蛋白质序列
def trim_sequence(sequence):
    if len(sequence) > 500:
        return sequence[:500]
    else:
        return sequence + 'X' * (500 - len(sequence))
        # return sequence

# 对第二列中的蛋白质序列进行修剪
df['Protein'] = df['Protein'].apply(trim_sequence)

# 重新计算修剪后的序列长度
df['Sequence Length'] = df['Protein'].apply(len)

# 将结果保存到新的Excel文件中
df.to_excel("./DrugBank_datasets/seed/train_output.xlsx", index=False)