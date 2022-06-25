from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import fix_seed
from annlp import ptm_path
import torch
from utils import random_mask
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path = ptm_path('roberta')
tokenizer = BertTokenizer.from_pretrained(path)


class BaseDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['source'])


def load_data(file_name, batch_size):
    df = pd.read_csv(file_name)
    encoding = tokenizer(text=df['text'].tolist(),
                         return_tensors='np',
                         truncation=True,
                         padding='max_length',
                         max_length=10)
    sources = []
    targets = []
    for input_ids in encoding['input_ids']:
        source, target = random_mask(input_ids, tokenizer)
        sources.append(source)
        targets.append(target)

    data = {'source': torch.Tensor(sources),
            'attention_mask': encoding['attention_mask'],
            'target': torch.Tensor(targets)}
    data_loader = DataLoader(BaseDataset(data),
                             batch_size,
                             pin_memory=True if torch.cuda.is_available() else False,
                             shuffle=False)
    return data_loader


# 训练模型
def train():
    fix_seed()

    train_data_loader = load_data('../data/tnews_public/train.csv', batch_size=32)
    dev_data_loader = load_data('../data/tnews_public/dev.csv', batch_size=32)

    model = BertForMaskedLM.from_pretrained(path)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

    min_loss = 1000
    for epoch in range(5):
        print('epoch:', epoch + 1)
        pbar = tqdm(train_data_loader)
        for data in pbar:
            optimizer.zero_grad()

            input_ids = data['source'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['target'].to(device).long()
            outputs = model(input_ids.long(), attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')

        dev_loss = 0
        for data in tqdm(dev_data_loader):
            input_ids = data['source'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['target'].to(device).long()
            with torch.no_grad():
                outputs = model(input_ids.long(), attention_mask=attention_mask, labels=labels)
            dev_loss += outputs.loss.item()
        print('dev loss:', dev_loss / len(dev_data_loader))
        print()

        if min_loss > dev_loss:
            min_loss = dev_loss
            torch.save(model, 'model.bin')


if __name__ == '__main__':
    train()
