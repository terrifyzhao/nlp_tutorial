from transformers import BertForMaskedLM, BertTokenizer, \
    BertPreTrainedModel, BertForSequenceClassification, BertModel
import torch
import pandas as pd
from utils import random_mask
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

path = 'E:\\ptm\\roberta'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForMaskedLM.from_pretrained(path)
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(path)


class BaseDataSet(Dataset):
    def __init__(self, encoding):
        self.encoding = encoding

    def __len__(self):
        return len(self.encoding['source'])

    def __getitem__(self, ids):
        item = {k: v[ids] for k, v in self.encoding.items()}
        return item


def load_data(file_name, batch_size):
    df = pd.read_csv(file_name)
    encoding = tokenizer(df['text'].tolist(),
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

    data_loader = DataLoader(BaseDataSet(data), batch_size=batch_size)
    return data_loader


def train():
    bs = 32
    train_data = load_data('../data/tnews_public/train.csv', batch_size=bs)
    dev_data = load_data('../data/tnews_public/dev.csv', batch_size=bs)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

    for epoch in range(3):
        pbar = tqdm(train_data)
        for data in pbar:
            optimizer.zero_grad()

            input_ids = data['source'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['target'].to(device)

            outputs = model(input_ids.long(), attention_mask, labels=labels.long())

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')

        dev_loss = 0
        for data in tqdm(dev_data):
            input_ids = data['source'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['target'].to(device)
            with torch.no_grad():
                outputs = model(input_ids.long(), attention_mask, labels=labels.long())
            dev_loss += outputs.loss.item()
        print('dev loss:', dev_loss / len(dev_data))

        torch.save(model, 'model.bin')


if __name__ == '__main__':
    train()
