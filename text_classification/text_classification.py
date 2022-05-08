import torch
from torch import nn
import jieba
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from utils import fix_seed


class TextCLS(torch.nn.Module):
    # 准备我们需要用到的参数和layer
    def __init__(self,
                 vocab_size,
                 embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # [batch_size, seq_len, hidden_size]
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True)
        self.dense1 = nn.Linear(256, 100)
        self.dense2 = nn.Linear(100, 2)

    # 前向传播，那我们准备好的layer拼接在一起
    def forward(self, x):
        embedding = self.embedding(x)
        # [batch_size, seq_len, hidden_size]
        out, _ = self.lstm(embedding)
        out = self.dense1(out[:, -1, :])
        out = self.dense2(out)
        return out


def tokenize(string):
    res = list(jieba.cut(string, cut_all=False))
    return res


# 把数据转换成index
def seq2index(seq, vocab):
    seg = tokenize(seq)
    seg_index = []
    for s in seg:
        seg_index.append(vocab.get(s, 1))
    return seg_index


# 统一长度
def padding_seq(X, max_len=10):
    return np.array([
        np.concatenate([x, [0] * (max_len - len(x))]) if len(x) < max_len else x[:max_len] for x in X
    ])


def load_data(batch_size=32):
    train_text = []
    train_label = []
    with open('../data/sentiment/sentiment.train.data', encoding='utf-8')as file:
        for line in file.readlines():
            t, l = line.strip().split('\t')
            train_text.append(t)
            train_label.append(int(l))

    dev_text = []
    dev_label = []
    with open('../data/sentiment/sentiment.valid.data', encoding='utf-8')as file:
        for line in file.readlines():
            t, l = line.strip().split('\t')
            dev_text.append(t)
            dev_label.append(int(l))

    # 生成词典
    segment = [tokenize(t) for t in train_text]

    word_frequency = defaultdict(int)
    for row in segment:
        for i in row:
            word_frequency[i] += 1

    word_sort = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)  # 根据词频降序排序

    vocab = {'[PAD]': 0, '[UNK]': 1}
    for d in word_sort:
        vocab[d[0]] = len(vocab)

    train_x = padding_seq([seq2index(t, vocab) for t in train_text])
    train_y = np.array(train_label)
    train_data_set = TensorDataset(torch.from_numpy(train_x),
                                   torch.from_numpy(train_y))
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=batch_size)

    dev_x = padding_seq([seq2index(t, vocab) for t in dev_text])
    dev_y = np.array(dev_label)
    dev_data_set = TensorDataset(torch.from_numpy(dev_x),
                                 torch.from_numpy(dev_y))
    dev_data_loader = DataLoader(dataset=dev_data_set, batch_size=batch_size)

    return train_data_loader, dev_data_loader, vocab


# 训练模型
def train():
    fix_seed()

    train_data_loader, dev_data_loader, vocab = load_data(128)
    model = TextCLS(vocab_size=len(vocab),
                    embedding_size=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()

    for epoch in range(5):
        print('epoch:', epoch + 1)
        pred = []
        label = []
        for step, (b_x, b_y) in enumerate(train_data_loader):
            if torch.cuda.is_available():
                b_x = b_x.cuda().long()
                b_y = b_y.cuda().long()
            output = model(b_x)
            pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            label.extend(b_y.cpu().numpy())
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            # 求解梯度
            loss.backward()
            # 更新我们的权重
            optimizer.step()
        acc = accuracy_score(pred, label)
        print('train acc:', acc)

        pred = []
        label = []
        for step, (b_x, b_y) in enumerate(dev_data_loader):
            if torch.cuda.is_available():
                b_x = b_x.cuda().long()
                b_y = b_y.cuda().long()
            with torch.no_grad():
                output = model(b_x)
            pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            label.extend(b_y.cpu().numpy())
        acc = accuracy_score(pred, label)
        print('dev acc:', acc)
        print()


if __name__ == '__main__':
    train()
