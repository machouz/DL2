import numpy as np


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
