import numpy as np
import torch
import dgl


# x是一个列表，其中每个元素是一个三元组(d, p, y)
# d是一个图; p是一个列表,其中包含与图中节点对应的属性; y是一个标签
def graph_collate_func(x):
    d, p, y = zip(*x)  # 将x压缩成三个单独的列表
    # d = dgl.batch(d)  # 将多个图合并成一个图批次
    return d, p, torch.tensor(y)


def loadProtT5andMol(train_num_batches, val_num_batches, test_num_batches):
    prott5_train_list = []
    prott5_val_list = []
    prott5_test_list = []

    mol_train_list = []
    mol_val_list = []
    mol_test_list = []

    dismap_train_list=[]
    dismap_val_list=[]
    dismap_test_list=[]

    for index in range(train_num_batches):
        # tmp1大小为(64,600,1280)
        tmp1 = np.load(f'../ProtT5_embedding/biosnap/seed/train/train_pro_emd_{index + 1}.npy')
        # tmp2大小为(64,290,768)
        tmp2 = np.load(f'../MolFormer_embedding/biosnap/seed/train/train_drug_emd_{index + 1}.npy')
        tmp3 = np.load(f'../dismap_adj/biosnap/seed/train/train_pro_adj_{index+1}.npy')
        tmp3 = tmp3[:-1, :-1]
        prott5_train_list.append(tmp1)
        mol_train_list.append(tmp2)
        dismap_train_list.append(tmp3)

    for index in range(val_num_batches):
        tmp1 = np.load(f'../ProtT5_embedding/biosnap/seed/val/val_pro_emd_{index + 1}.npy')
        tmp2 = np.load(f'../MolFormer_embedding/biosnap/seed/val/val_drug_emd_{index + 1}.npy')
        tmp3 = np.load(f'../dismap_adj/biosnap/seed/val/val_pro_adj_{index + 1}.npy')
        tmp3 = tmp3[:-1, :-1]
        prott5_val_list.append(tmp1)
        mol_val_list.append(tmp2)
        dismap_val_list.append(tmp3)

    for index in range(test_num_batches):
        tmp1 = np.load(f'../ProtT5_embedding/biosnap/seed/test/test_pro_emd_{index + 1}.npy')
        tmp2 = np.load(f'../MolFormer_embedding/biosnap/seed/test/test_drug_emd_{index + 1}.npy')
        tmp3 = np.load(f'../dismap_adj/biosnap/seed/test/test_pro_adj_{index + 1}.npy')
        tmp3 = tmp3[:-1, :-1]
        prott5_test_list.append(tmp1)
        mol_test_list.append(tmp2)
        dismap_test_list.append(tmp3)

    return (prott5_train_list, prott5_val_list, prott5_test_list, mol_train_list, mol_val_list, mol_test_list
            ,dismap_train_list,dismap_val_list,dismap_test_list)