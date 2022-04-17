from elmoformanylangs import Embedder
import jieba

sentence = '我爱自然语言处理'

segment = list(jieba.cut(sentence))
print(segment)
model = Embedder('../ptm/elmo')
vec = model.sents2elmo([segment])
print(vec)
