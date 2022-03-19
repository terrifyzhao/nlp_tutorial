import numpy as np
import gensim

model = gensim.models.Word2Vec.load('word2vec/wiki.model')
embedding = model.wv


def cosine(a, b):
    return np.matmul(a, b.T) / np.linalg.norm(a) / np.linalg.norm(b, axis=-1)


def search(word, topk=3):
    we = embedding[word]
    similarity = cosine(we, embedding.vectors)
    index = np.argsort(-similarity)
    w = np.array(embedding.index2word)[index[0:topk]]
    print(w)


if __name__ == '__main__':
    while 1:
        text = input('word:')
        search(text)
