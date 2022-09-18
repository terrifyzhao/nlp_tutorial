import jieba
import numpy as np
import os
import random
import torch


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


def fix_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def random_mask(input_ids, tokenizer):
    length = len(input_ids)
    # 移除pad cls sep
    input_ids = input_ids[1:-1]
    prob = np.random.random(len(input_ids))
    source, target = [], []
    # cls:[101]
    source.append(101)
    target.append(-100)
    # p->[0:1]
    for p, ids in zip(prob, input_ids):
        if p < 0.15 * 0.8:
            source.append(tokenizer.mask_token_id)
            target.append(ids)
        elif p < 0.15 * 0.9:
            source.append(ids)
            target.append(ids)
        elif p < 0.15:
            source.append(np.random.choice(tokenizer.vocab_size))
            target.append(ids)
        else:
            source.append(ids)
            target.append(-100)
    # sep:[102]
    source.append(102)
    target.append(-100)
    while len(source) < length:
        source.append(0)
        target.append(-100)
    return source, target


def punctuation():
    import string
    en_punctuation = list(string.punctuation)
    zh_punctuation = ['，', '。', '：', '！', '？', '《', '》', '"', '；', "'"]
    return en_punctuation + zh_punctuation


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def one_hot(x, n_class):
    return torch.nn.functional.one_hot(x, num_classes=n_class)
