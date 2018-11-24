import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
from Ass2.utils import *

train_name = sys.argv[1]
LR = 0.01
EPOCHS = 10
train = np.loadtxt(train_name, np.str)
words_id = {word: i for i, word in enumerate(list(set(train[:, 0])) + ["*UNK*"])}
label_id = {label: i for i, label in enumerate(set(train[:, 1]))}
id_label = {i: label for label, i in label_id.items()}


class MLP(nn.Module):
    def __init__(self, vocab_size, output_layer, embedding_dim=50, window_size=5, hidden_layer=100):
        super(MLP, self).__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc0 = nn.Linear(embedding_dim * window_size, hidden_layer)
        self.fc1 = nn.Linear(hidden_layer, output_layer)

    def forward(self, data):
        input = self.embeddings(data).view(-1, self.embedding_dim * self.window_size)
        out = self.fc0(input)
        out = F.tanh(out)
        out = self.fc1(out)
        out = F.softmax(out, dim=1)
        return out


def train_model(model, optimizer, train_data, batch_size):
    model.train()
    for i in xrange(0, len(train_data), batch_size):
        data = train_data[i:i + batch_size, :-1]
        label = train_data[i:i + batch_size, -1]
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()


def test_model(model, test_file, batch_size=None):
    model.eval()
    loss = 0
    correct = 0
    count = 0
    test = np.loadtxt(test_file, np.str)
    vecs = np.array(map(lambda (word, tag): [words_id[word], label_id[tag]], test))
    test_data = torch.LongTensor(
        zip(vecs[:, 0], vecs[1:, 0], vecs[2:, 0], vecs[3:, 0], vecs[4:, 0],
            vecs[2:, 1]))
    if batch_size == None:
        batch_size = len(test_data)
    for i in xrange(0, len(test_data), batch_size):
        data = train_data[i:i + batch_size, :-1]
        label = train_data[i:i + batch_size, -1]
        output = model(data)
        loss += F.cross_entropy(output, label)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label).cpu().sum()
        count += len(test_data)

    loss /= count
    accuracy = correct / count
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, count, 100. * correct / count))
    return loss, accuracy


def predict(model, fname):
    data = np.loadtxt(fname, np.str)
    vecs = np.array(map(lambda (word, tag): [words_id[word], label_id[tag]], data))
    input = torch.LongTensor(zip(vecs[:, 0], vecs[1:, 0], vecs[2:, 0], vecs[3:, 0], vecs[4:, 0]))
    output = model(input)
    pred = output.data.max(1, keepdim=True)[1]
    return pred


if __name__ == '__main__':

    num_words = len(words_id)
    train_vecs = np.array(map(lambda (word, tag): [words_id[word], label_id[tag]], train))

    model = MLP(vocab_size=num_words, output_layer=len(label_id))
    train_data = torch.LongTensor(
        zip(train_vecs[:, 0], train_vecs[1:, 0], train_vecs[2:, 0], train_vecs[3:, 0], train_vecs[4:, 0],
            train_vecs[2:, 1]))
    optimizer = optim.SGD(model.parameters(), lr=LR)

    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        train_model(model, optimizer, train_data, 1000)
    test_model(model, train_name, 10000)
