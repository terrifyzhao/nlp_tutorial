import torch
from torch import nn
import jieba
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from utils import fix_seed
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import fix_seed
import json
from annlp import ptm_path, print_sentence_length
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch import softmax

path = 'E:\\ptm\\roberta'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained(path)


class TextCLS(torch.nn.Module):
    # 准备我们需要用到的参数和layer
    def __init__(self,
                 embedding_size,
                 vocab_size=21128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # [batch_size, seq_len, hidden_size]
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True)
        self.dense1 = nn.Linear(256, 100)
        self.dense2 = nn.Linear(100, 4)

    # 前向传播，那我们准备好的layer拼接在一起
    def forward(self, x):
        embedding = self.embedding(x)
        # [batch_size, seq_len, hidden_size]
        out, _ = self.lstm(embedding)
        out = self.dense1(out[:, -1, :])
        out = self.dense2(out)
        return out


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


def load_data(batch_size=32):
    train_df = pd.read_csv('../data/tnews_public/train.csv')
    train_text = train_df['text'].tolist()
    train_label = train_df['label'].tolist()
    train_text = tokenizer(text=train_text,
                           return_tensors='pt',
                           truncation=True,
                           padding=True,
                           max_length=32)
    train_loader = DataLoader(BaseDataset(train_text, train_label),
                              batch_size,
                              pin_memory=True if torch.cuda.is_available() else False,
                              shuffle=False)

    dev_df = pd.read_csv('../data/tnews_public/dev.csv')
    dev_text = dev_df['text'].tolist()
    dev_label = dev_df['label'].tolist()
    dev_text = tokenizer(text=dev_text,
                         return_tensors='pt',
                         truncation=True,
                         padding=True,
                         max_length=32)
    dev_loader = DataLoader(BaseDataset(dev_text, dev_label),
                            batch_size,
                            pin_memory=True if torch.cuda.is_available() else False,
                            shuffle=False)

    return train_loader, dev_loader


def CE(pred, label, t=1):
    pred = softmax(pred / t, dim=-1)
    label = softmax(label / t, dim=-1)
    loss = -torch.sum(torch.log(pred) * label)
    return loss


# 训练模型
def train():
    fix_seed()

    train_data_loader, dev_data_loader = load_data(32)
    student = TextCLS(embedding_size=100)
    student = student.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(20):
        print('epoch:', epoch + 1)
        pred = []
        label = []
        pbar = tqdm(train_data_loader)
        for data in pbar:
            optimizer.zero_grad()

            input_ids = data['input_ids'].to(device)
            labels = data['labels'].to(device)

            output = student(input_ids)
            loss = loss_func(output, labels)

            pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            label.extend(labels)
            loss.backward()

            optimizer.step()

            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')

        pred = []
        label = []
        for data in tqdm(dev_data_loader):
            input_ids = data['input_ids'].to(device)
            labels = data['labels'].to(device).long()
            with torch.no_grad():
                output = student(input_ids)
            pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            label.extend(labels.cpu().numpy())
        acc = accuracy_score(pred, label)
        print('dev acc:', acc)
        print()
        if acc > best_acc:
            torch.save(student.state_dict(), 'model.bin')
            best_acc = acc


if __name__ == '__main__':
    train()