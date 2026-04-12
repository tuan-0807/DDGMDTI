from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import pandas as pd

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 加载模型和分词器，启用半精度（FP16）减少显存占用
tokenizer = T5Tokenizer.from_pretrained('../model/protT5_model', do_lower_case=False)
model = T5EncoderModel.from_pretrained("../model/protT5_model").to(device).half()  # 半精度
model.eval()  # 推理模式，关闭 dropout 等，减少显存波动

# 加载数据
df = pd.read_excel('../bioSNAP_datasets/cluster/target_train_output.xlsx')
protein_sequences_total = df['Protein'].tolist()


batch_size = 64

index = 1
for i in tqdm(range(0, len(protein_sequences_total), batch_size)):
    # 取当前批次序列
    sequence_examples = protein_sequences_total[i:i + batch_size]

    # 序列预处理（同原逻辑）
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():  # 关闭梯度计算，减少显存
        embedding_rpr = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

    # 提取最后一层注意力图（第23层），并平均注意力头
    atten_map_lastlayer = embedding_rpr.attentions[23][0]  # [num_heads, seq_len, seq_len]
    atten_map_lastlayer = torch.mean(atten_map_lastlayer, dim=0)  # 平均所有头 [seq_len, seq_len]

    # 4. 转移到CPU再保存（释放GPU显存）
    atten_map_lastlayer = atten_map_lastlayer.cpu().numpy()
    np.save(f'../../Mol+ProtT5/dismap_adj/bioSNAP/cluster/target_val/val_pro_adj_{index}.npy', atten_map_lastlayer)

    # 5. 手动释放当前批次的GPU显存
    del input_ids, attention_mask, embedding_rpr, atten_map_lastlayer
    torch.cuda.empty_cache()  # 清空缓存

    index += 1










