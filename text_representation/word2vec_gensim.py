from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

input_file = 'word2vec/wiki.txt'
out_file = 'word2vec/wiki.model'

model = Word2Vec(LineSentence(input_file),
                 size=100,
                 window=5,
                 min_count=5,
                 workers=multiprocessing.cpu_count(),
                 sg=1,
                 hs=0,
                 negative=5)

model.save(out_file)
