import numpy as np


def write_to_file(fname, data):
    np.savetxt(fname, data, fmt="%s", delimiter='\n')


def dic_to_file(dic, fname):
    data = []
    for key, label in dic.items():
        data.append(key + "\t" + str(label))
    write_to_file(fname, data)


def create_id(fname):
    label_id = {}
    id_label = {}
    data = np.loadtxt(fname, np.str)
    i = 0
    for word_tag in data:
        label = word_tag[1]
        if label not in label_id:
            label_id[label] = i
            id_label[i] = label
            i += 1
    return label_id, id_label
