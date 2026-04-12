import torch.utils.data as data

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs  # 是一个数组，表示drug-protain的序号
        self.df = df
        self.max_drug_nodes = max_drug_nodes


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]  # 通过索引获取样本ID
        v_d = self.df.iloc[index]['SMILES']  # 获取索引为index的药物的SMILES序列
        v_p = self.df.iloc[index]['Protein']  # 获取对应索引的蛋白质序列
        # v_p = self.df.iloc[index]['Protein_600']  # 获取对应索引的蛋白质序列


        # 下面处理标签部分
        y = self.df.iloc[index]["Y"]
        return v_d, v_p, y
