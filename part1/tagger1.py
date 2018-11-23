import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
from Ass2.utils import *

train_name = sys.argv[1]
LR = 0.01
EPOCHS = 10


class MLP(nn.Module):
    def __init__(self, vocab_size, output_layer, embedding_dim=50, window_size=5, hidden_layer=100):
        super(MLP, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc0 = nn.Linear(embedding_dim * window_size, hidden_layer)
        self.fc1 = nn.Linear(hidden_layer, output_layer)

    def forward(self, arr):
        lst = []
        for x in arr:
            x = torch.LongTensor([x])
            embeds = self.embeddings(x).view((1, -1))
            lst.append(embeds)
        input = torch.cat(lst, 1)
        out = self.fc0(input)
        out = F.tanh(out)
        out = self.fc1(out)
        out = F.log_softmax(out, dim=1)
        return out


def train_model(model, optimizer, train_data):
    model.train()
    for word in train_data:
        data = word[:-1]
        label = word[-1]
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    train = np.loadtxt(train_name, np.str)
    words_id, id_words = create_id(train[:, 0])
    label_id, id_label = create_id(train[:, 1])
    num_words = len(words_id)
    train_vecs = np.array(map(lambda (word, tag): [words_id[word], label_id[tag]], train))

    model = MLP(vocab_size=num_words, output_layer=len(label_id))
    train_data = zip(train_vecs[:, 0], train_vecs[1:, 0], train_vecs[2:, 0], train_vecs[3:, 0], train_vecs[4:, 0],
                     train_vecs[2:, 1])
    optimizer = optim.SGD(model.parameters(), lr=LR)
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        train_model(model, optimizer, train_data)
