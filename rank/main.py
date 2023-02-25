import pandas as pd
from utils import cos_sim
import numpy as np
import jieba
from collections import Counter
from rank import rank
from text_representation.sentence_embedding import SentenceEmbedding

model = SentenceEmbedding()
data = pd.read_csv('../data/rank/qa_data.csv')
question = data['question'].values
embedding = model.encode(data['question'].tolist())


class BM25:
    def __init__(self, documents_list, k1=2, k2=1, b=0.75):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                    self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                             qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        query = list(jieba.cut(query))
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list


bm = BM25(question)


def word_recall(text):
    score = bm.get_documents_score(text)
    index = np.argsort(-np.array(score))[:10]
    candidate = question[index]
    return candidate


def embedding_recall(text):
    e = model.encode(text)
    sim = cos_sim(e, embedding)[0]
    index = np.argsort(-sim)[:10]
    candidate = question[index]
    return candidate


if __name__ == '__main__':

    while 1:
        text = input('text:')
        res1 = list(embedding_recall(text))
        print(res1)
        res2 = list(word_recall(text))
        print(res2)
        res1.extend(res2)
        recall_data = list(set(res1))
        rank(text, recall_data)
