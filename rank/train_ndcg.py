import numpy as np
from transformers import BertTokenizer
from model import BertForNDCG
import pandas as pd
import torch
from annlp import ptm_path, get_device
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


def dcg(score):
    index = list(range(1, len(score[0]) + 1))
    return score[:, 0] + np.sum(score[:, 1:] / np.log2(index[1:]), axis=1)


def ndcg(score):
    if not isinstance(score, np.ndarray):
        score = np.array(score)
    if score.ndim == 1:
        score = score[None, :]
    dcg_score = dcg(score)
    idcg_score = dcg(np.sort(score[0][None, :])[:, ::-1])
    ndcg_socre = dcg_score / idcg_score
    return ndcg_socre


class BaseDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


batch_size = 20
device = get_device()
path = 'E:\\ptm\\roberta'
# path = ptm_path('roberta')
tokenizer = BertTokenizer.from_pretrained(path)
model = BertForNDCG.from_pretrained(path).to(device)

data = pd.read_csv('../data/rank/sort_data.csv')
text = data['text'].tolist()
all_text = []
query = []
document = []

q = ''
for i, t in enumerate(text):
    if i % 5 == 0:
        q = t
    query.append(q)
    document.append(t)
encoding = tokenizer(query[:-1000], document[:-1000], truncation=True, padding=True, max_length=128, return_tensors='pt')
encoding_dev = tokenizer(query[-1000:], document[-1000:], truncation=True, padding=True, max_length=64,
                         return_tensors='pt')

train_loader = DataLoader(BaseDataset(encoding), batch_size=batch_size)
dev_loader = DataLoader(BaseDataset(encoding_dev), batch_size=batch_size)


def dev_func():
    model.eval()
    all_ndcg = []
    with torch.no_grad():
        for data in tqdm(dev_loader):
            outputs = model(input_ids=data['input_ids'].to(device),
                            attention_mask=data['attention_mask'].to(device),
                            token_type_ids=data['token_type_ids'].to(device),
                            num=5)
            logits = outputs[1]
            score = torch.argsort(logits) + 1
            a = ndcg(score.cpu().numpy())
            all_ndcg.extend(a)
    ndcg_score = np.mean(all_ndcg)
    print('ndcg:', ndcg_score)
    return ndcg_score


opt = torch.optim.Adam(lr=5e-5, params=model.parameters())
best_ndcg = 0
for epoch in range(10):
    model.train()
    pbar = tqdm(train_loader)
    for data in pbar:
        opt.zero_grad()
        outputs = model(input_ids=data['input_ids'].to(device),
                        attention_mask=data['attention_mask'].to(device),
                        token_type_ids=data['token_type_ids'].to(device),
                        num=5)
        loss, score = outputs[0], outputs[1]
        loss.backward()
        opt.step()

        pbar.update()
        pbar.set_description(f'loss:{loss.item():.4f}')

    cur_ndcg = dev_func()
    if cur_ndcg > best_ndcg:
        best_ndcg = cur_ndcg
        torch.save(model.state_dict(), 'best_model.bin')
