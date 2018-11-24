import numpy as np
import sys
from numpy import linalg as LA

vocabFile = sys.argv[1]
vectorsFile = sys.argv[2]

def normalize(vec):
    norme = LA.norm(vec)
    return vec/norme

vocab = []
for line in file(vocabFile):
    vocab.append(line[:-1])

vocab = np.array(vocab)
wordVectors = np.loadtxt(vectorsFile)
wordVectors = np.array(map(lambda x: normalize(x), wordVectors))

W2i = {word: i for i, word in enumerate(vocab) }

def most_similar(word, k):
    word_vec = wordVectors[W2i[word]]
    w = wordVectors.dot(word_vec)
    sims = w.argsort()[::-1][:k + 1]
    return zip(vocab[sims[1:]], w[sims[1:]])


