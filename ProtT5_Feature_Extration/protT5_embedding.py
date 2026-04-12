from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np
import pandas as pd
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('./model/protT5_model', do_lower_case=False, legacy=False)
model = T5EncoderModel.from_pretrained("./model/protT5_model").to(device)

df = pd.read_excel('./bioSNAP_datasets/length600/test.xlsx')
protein_sequences_total = df['Protein_600'].tolist()

# import pdb
# pdb.set_trace()

index = 1
for i in tqdm(range(0, len(protein_sequences_total), 64)):

    sequence_examples = protein_sequences_total[i:i+64]
    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # 对氨基酸进行编码
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
    # 对每个氨基酸编码，并在最右边补一个
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings 大小为(64,601,1024)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)


    # 获取去掉特殊字符后的嵌入表示 大小为(64,600,1024)
    filtered_embeddings = np.array(embedding_repr.last_hidden_state[:, :-1, :].cpu())
    np.save(f'./bioSNAP_pro_emd/length600/test/test_pro_emd_{index}.npy', filtered_embeddings)
    index += 1




