import jieba
import numpy as np


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
