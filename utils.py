import numpy as np


def load_train(fname):
    begin_tag = "STR"
    begin_word = "***"
    data = [begin_word, begin_word]
    tags = [begin_tag, begin_tag]
    in_file = file(fname, 'r')
    for line in in_file:
        splitted_data = line.rsplit()
        if len(splitted_data) == 0:
            data.append(begin_word)
            tags.append(begin_tag)
            data.append(begin_word)
            tags.append(begin_tag)
        else:
            word, tag = splitted_data
            data.append(word)
            tags.append(tag)
    return data, tags


def load_test(fname):
    begin_word = "***"
    data = [begin_word, begin_word]
    in_file = file(fname, 'r')
    for line in in_file:
        splitted_data = line.rsplit()
        if len(splitted_data) == 0:
            data.append(begin_word)
            data.append(begin_word)
        else:
            word, tag = splitted_data
            data.append(word)
    return data


def write_to_file(fname, data):
    np.savetxt(fname, data, fmt="%s", delimiter='\n')


def dic_to_file(dic, fname):
    data = []
    for key, label in dic.items():
        data.append(key + "\t" + str(label))
    write_to_file(fname, data)


def create_id(vec):
    element_id = {}
    id_element = {}
    s = set(vec)
    i = 0
    for element in s:
        element_id[element] = i
        id_element[i] = element
        i += 1
    return element_id, id_element
