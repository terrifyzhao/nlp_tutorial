import torch
from torch import nn
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from text_similarity.esim import ESIM
from text_similarity.dssm import DSSM
from utils import *
import pandas as pd
from tqdm import tqdm


def load_data(batch_size=32):
    train = pd.read_csv('../data/LCQMC/lcqmc_train.csv')
    dev = pd.read_csv('../data/LCQMC/lcqmc_dev.csv')

    text = train['sentence1'].tolist()
    text.extend(train['sentence2'].tolist())
    text.extend(dev['sentence1'].tolist())
    text.extend(dev['sentence2'].tolist())

    # 生成词典
    segment = [tokenize(t) for t in text]

    word_frequency = defaultdict(int)
    for row in segment:
        for i in row:
            word_frequency[i] += 1

    word_sort = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)  # 根据词频降序排序

    vocab = {'[PAD]': 0, '[UNK]': 1}
    for d in word_sort:
        vocab[d[0]] = len(vocab)

    train_x1 = padding_seq([seq2index(t, vocab) for t in train['sentence1'].tolist()])
    train_x2 = padding_seq([seq2index(t, vocab) for t in train['sentence2'].tolist()])
    train_data_set = TensorDataset(torch.from_numpy(train_x1),
                                   torch.from_numpy(train_x2),
                                   torch.from_numpy(train['label'].values))
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=batch_size)

    dev_x1 = padding_seq([seq2index(t, vocab) for t in dev['sentence1'].tolist()])
    dev_x2 = padding_seq([seq2index(t, vocab) for t in dev['sentence2'].tolist()])
    dev_data_set = TensorDataset(torch.from_numpy(dev_x1),
                                 torch.from_numpy(dev_x2),
                                 torch.from_numpy(dev['label'].values))
    dev_data_loader = DataLoader(dataset=dev_data_set, batch_size=batch_size)

    return train_data_loader, dev_data_loader, vocab


# 训练模型
def train():
    train_data_loader, dev_data_loader, vocab = load_data(32)
    # ptm = ESIM(char_vocab_size=len(vocab),
    #              char_dim=100,
    #              char_hidden_size=128,
    #              hidden_size=128,
    #              max_word_len=10)
    model = DSSM(char_vocab_size=len(vocab),
                 char_dim=100,
                 hidden_size=128,
                 max_word_len=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.BCELoss()

    if torch.cuda.is_available():
        model = model.cuda()

    for epoch in range(5):
        pred = []
        label = []
        for step, (x1, x2, y) in tqdm(enumerate(train_data_loader)):
            if torch.cuda.is_available():
                x1 = x1.cuda().long()
                x2 = x2.cuda().long()
                y = y.cuda()
            output = model(x1, x2)

            pred.extend((output.cpu().data.numpy() > 0.5).astype(int))
            label.extend(y.cpu().numpy())
            loss = loss_func(output, y.float())
            optimizer.zero_grad()
            # 求解梯度
            loss.backward()
            # 更新我们的权重
            optimizer.step()
        acc = accuracy_score(pred, label)
        print('train acc:', acc)

        pred = []
        label = []
        for step, (x1, x2, y) in tqdm(enumerate(dev_data_loader)):
            if torch.cuda.is_available():
                x1 = x1.cuda().long()
                x2 = x2.cuda().long()
                y = y.cuda()
            with torch.no_grad():
                output = model(x1, x2)
            pred.extend((output.cpu().data.numpy() > 0.5).astype(int))
            label.extend(y.cpu().numpy())
        acc = accuracy_score(pred, label)
        print('dev acc:', acc)


if __name__ == '__main__':
    train()
