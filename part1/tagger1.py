import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
from Ass2.utils import *

USE_PRETRAINED = True if len(sys.argv) > 3 else False
print "Using pretrained" if USE_PRETRAINED else "Not using pretrained"
train_name = sys.argv[1]  # "data/pos/train"
dev_name = sys.argv[2]  # "data/pos/dev"
print "Using train: " + train_name
print "Using dev: " + dev_name

vocabFile = sys.argv[3] if USE_PRETRAINED else None
vectorsFile = sys.argv[4] if USE_PRETRAINED else None

LR = 0.01
LR_DECAY = 0.8
EPOCHS = 10
BATCH_SIZE = 1000
HIDDEN_LAYER = 150

words, labels = load_train(train_name)
if USE_PRETRAINED:
    vocab = []
    for line in file(vocabFile):
        vocab.append(line[:-1])
    vocab = np.array(vocab)
    words_id = {word: i for i, word in enumerate(vocab)}
    wordVectors = np.loadtxt(vectorsFile)
    wordVectors = np.array(map(lambda x: x / np.linalg.norm(x), wordVectors))
else:
    words_id = {word: i for i, word in enumerate(list(set(words)) + ["UUUNKKK"])}
label_id = {label: i for i, label in enumerate(set(labels))}
id_label = {i: label for label, i in label_id.items()}


def get_words_id(word, words_id=words_id):
    if USE_PRETRAINED:
        word = word.lower()
    if word not in words_id:
        return words_id["UUUNKKK"]
    return words_id[word]


class MLP(nn.Module):
    def __init__(self, output_layer, vocab_size=None, pre_trained_vec=None, embedding_dim=50, window_size=5,
                 hidden_layer=100):
        super(MLP, self).__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        if USE_PRETRAINED:
            pre_trained_vec = torch.FloatTensor(pre_trained_vec)
            self.embeddings = nn.Embedding.from_pretrained(pre_trained_vec, freeze=False)
        else:
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
    loss_history = []
    for i in xrange(0, len(train_data), batch_size):
        data = train_data[i:i + batch_size, :-1]
        label = train_data[i:i + batch_size, -1]
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss_history.append(loss)
        loss.backward()
        optimizer.step()
    return loss_history


def test_model(model, test_file):
    model.eval()
    loss = correct = count = 0
    test_words, test_labels = load_train(test_file)
    vecs = np.array(map(lambda (word, tag): [get_words_id(word), label_id[tag]], zip(test_words, test_labels)))
    test_data = torch.LongTensor(
        zip(vecs[:, 0], vecs[1:, 0], vecs[2:, 0], vecs[3:, 0], vecs[4:, 0],
            vecs[2:, 1]))
    for i in xrange(0, len(test_data), 1):
        data = test_data[i:i + 1, :-1]
        labels = test_data[i:i + 1, -1]
        output = model(data)
        loss += F.cross_entropy(output, labels)
        pred = output.data.max(1, keepdim=True)[1].view(-1)
        if labels.item() != label_id['STR']:
            correct += (pred == labels).cpu().sum().item()
            count += 1

    accuracy = float(correct) / count
    print('Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, count, 100. * accuracy))
    return loss, accuracy


def test_ner_model(model, test_file):
    model.eval()
    loss = correct = count = 0
    test_words, test_labels = load_train(test_file)
    vecs = np.array(map(lambda (word, tag): [get_words_id(word), label_id[tag]], zip(test_words, test_labels)))
    test_data = torch.LongTensor(
        zip(vecs[:, 0], vecs[1:, 0], vecs[2:, 0], vecs[3:, 0], vecs[4:, 0],
            vecs[2:, 1]))
    for i in xrange(0, len(test_data), 1):
        data = test_data[i:i + 1, :-1]
        labels = test_data[i:i + 1, -1]
        output = model(data)
        loss += F.cross_entropy(output, labels)
        pred = output.data.max(1, keepdim=True)[1].view(-1)
        if labels.item() != label_id['O'] and labels.item() != label_id['STR']:
            correct += (pred == labels).cpu().sum().item()
            count += 1

    accuracy = float(correct) / count
    print('Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, count, 100. * accuracy))
    return loss, accuracy


def predict(model, fname, output_file="test1.ner"):
    data = load_test(fname)
    vecs = np.array(map(lambda word: get_words_id(word), data))
    input = torch.LongTensor(zip(vecs[:], vecs[1:], vecs[2:], vecs[3:], vecs[4:]))
    output = model(input)
    pred = output.data.max(1, keepdim=True)[1]
    e = []
    for i in pred.numpy():
        e.append(id_label[i[0]])
    np.savetxt(output_file, e, fmt="%s")
    return pred


if __name__ == '__main__':
    print('Learning rate {}'.format(LR))
    print('Learning rate decay {}'.format(LR_DECAY))
    print('Hidden layer {}'.format(HIDDEN_LAYER))
    print('Batch size {}'.format(BATCH_SIZE))

    train_vecs = np.array(map(lambda (word, tag): [get_words_id(word), label_id[tag]], zip(words, labels)))
    if USE_PRETRAINED:
        model = MLP(pre_trained_vec=wordVectors, output_layer=len(label_id), hidden_layer=HIDDEN_LAYER)
    else:
        model = MLP(vocab_size=len(words_id), output_layer=len(label_id), hidden_layer=HIDDEN_LAYER)
    train_data = torch.LongTensor(
        zip(train_vecs[:, 0], train_vecs[1:, 0], train_vecs[2:, 0], train_vecs[3:, 0], train_vecs[4:, 0],
            train_vecs[2:, 1]))
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=LR)

    loss_history = []
    accuracy_history = []
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        if epoch % 1 == 0:
            loss, accuracy = test_model(model, dev_name)
            loss_history.append(loss)
            accuracy_history.append(accuracy)
        train_model(model, optimizer, train_data, BATCH_SIZE)
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY
    create_graph("POS_loss", [loss_history], make_new=True)
    create_graph("POS_accuracy", [accuracy_history], ylabel="Accuracy", make_new=True)
