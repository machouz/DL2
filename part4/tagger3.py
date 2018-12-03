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
train_name = sys.argv[1]  # "data/pos/train"
dev_name = sys.argv[2]  # "data/pos/dev"
vocabFile = sys.argv[3] if USE_PRETRAINED else None
vectorsFile = sys.argv[4] if USE_PRETRAINED else None

LR = 0.01
LR_DECAY = 0.8
EPOCHS = 10
BATCH_SIZE = 10000
HIDDEN_LAYER = 150

words, labels = load_train(train_name)
if USE_PRETRAINED:
    vocab = []
    for line in file(vocabFile):
        vocab.append(line[:-1])
    wordVectors = np.loadtxt(vectorsFile)
    wordVectors = np.array(map(lambda x: x / np.linalg.norm(x), wordVectors))
else:
    vocab = ["UUUNKKK"] + list(set(words))

vocab_subwords = create_subwords(vocab)
words_id = {word: i for i, word in enumerate(vocab_subwords)}

label_id = {label: i for i, label in enumerate(set(labels))}
id_label = {i: label for label, i in label_id.items()}


def get_words_id(word, words_id=words_id):
    if word not in words_id:
        return words_id["UUUNKKK"]
    return words_id[word]


class MLP(nn.Module):
    def __init__(self, output_layer, vocab_size=None, pre_trained_vec=None, embedding_dim=50, window_size=5,
                 hidden_layer=100):
        super(MLP, self).__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if USE_PRETRAINED:
            pre_trained_vec = torch.FloatTensor(pre_trained_vec)
            emb = torch.cat((pre_trained_vec, self.embeddings.weight[len(pre_trained_vec):]))
            self.embeddings.weight = nn.Parameter(emb)
        self.fc0 = nn.Linear(embedding_dim * window_size, hidden_layer)
        self.fc1 = nn.Linear(hidden_layer, output_layer)

    def forward(self, data):
        input = self.embeddings(data)
        out = input.sum(dim=2)
        out = out.view(-1, self.embedding_dim * self.window_size)
        out = self.fc0(out)
        out = F.tanh(out)
        out = self.fc1(out)
        out = F.softmax(out, dim=1)
        return out


def train_model(model, optimizer, train_data, labels, batch_size):
    model.train()
    loss_history = []
    for i in xrange(0, len(train_data), batch_size):
        data = train_data[i:i + batch_size]
        label = torch.LongTensor(labels[i:i + batch_size])
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
    test_vecs = map(lambda word: [get_words_id(word), get_words_id(get_prefix(word)), get_words_id(get_suffix(word))],
            test_words)
    test_label = torch.LongTensor(map(lambda tag: label_id[tag], test_labels)[2:-2])
    test_data = torch.LongTensor(
        zip(test_vecs[:], test_vecs[1:], test_vecs[2:], test_vecs[3:], test_vecs[4:]))
    for i in xrange(0, len(test_data), 1):
        data = train_data[i:i + 1]
        labels = test_label[i:i + 1]
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
    test_vecs = map(lambda word: [get_words_id(word), get_words_id(get_prefix(word)), get_words_id(get_suffix(word))],
                    test_words)
    test_label = torch.LongTensor(map(lambda tag: label_id[tag], test_labels)[2:-2])
    test_data = torch.LongTensor(
        zip(test_vecs[:], test_vecs[1:], test_vecs[2:], test_vecs[3:], test_vecs[4:]))
    for i in xrange(0, len(test_data), 1):
        data = train_data[i:i + 1]
        labels = test_label[i:i + 1]
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
    print "Using pretrained" if USE_PRETRAINED else "Not using pretrained"
    print("Using train:  {}".format(train_name))
    print("Using dev:  {}".format(dev_name))
    print('Learning rate {}'.format(LR))
    print('Learning rate decay {}'.format(LR_DECAY))
    print('Hidden layer {}'.format(HIDDEN_LAYER))
    print('Batch size {}'.format(BATCH_SIZE))

    train_vecs = np.array(
        map(lambda word: [get_words_id(word), get_words_id(get_prefix(word)), get_words_id(get_suffix(word))], words))
    train_label = torch.LongTensor(map(lambda tag: label_id[tag], labels)[2:-2])
    train_data = torch.LongTensor(
        zip(train_vecs[:], train_vecs[1:], train_vecs[2:], train_vecs[3:], train_vecs[4:]))
    if USE_PRETRAINED:
        model = MLP(vocab_size=len(words_id), pre_trained_vec=wordVectors, output_layer=len(label_id),
                    hidden_layer=HIDDEN_LAYER)
    else:
        model = MLP(vocab_size=len(words_id), output_layer=len(label_id), hidden_layer=HIDDEN_LAYER)
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
        train_model(model, optimizer, train_data, train_label, BATCH_SIZE)
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY
    # create_graph("POS_loss", [loss_history], make_new=True)
    # create_graph("POS_accuracy", [accuracy_history], ylabel="Accuracy", make_new=True)
