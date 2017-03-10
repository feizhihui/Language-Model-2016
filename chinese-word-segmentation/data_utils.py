# encoding=utf-8
import collections
import numpy as np


def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    else:
        return False


def get_dictionary(data_path='F:/Datas/msr_training/msr_training.utf8.ic'):
    file = (open(data_path, encoding='utf-8').readlines())[1:]
    word_doc = []
    for line in file:
        word_doc.append(line[0])
    cnt = collections.Counter(word_doc).most_common()
    dictionary = {}
    reverse_dictionary = {}
    for i, t in enumerate(cnt):
        dictionary[t[0]] = i
        reverse_dictionary[i] = t[0]
    return dictionary, reverse_dictionary


def read_data(data_path='F:/Datas/msr_training/msr_training.utf8.ic', target_dict={'B': 0, 'M': 1, 'E': 2, 'S': 3}):
    file = (open(data_path, encoding='utf-8').readlines())[1:]
    word_doc = []
    labels = []
    for line in file:
        word_doc.append(line[0])
        labels.append(line[2])
    cnt = collections.Counter(word_doc).most_common()
    dictionary = {}
    reverse_dictionary = {}
    for i, t in enumerate(cnt):
        dictionary[t[0]] = i
        reverse_dictionary[i] = t[0]

    targets = []
    words = []
    for i, v in enumerate(labels):
        targets.append(target_dict[v])
        words.append(dictionary[word_doc[i]])
    return (np.array(words), targets)


def ctb_iterator(raw_data, batch_size, num_steps):
    data_len = len(raw_data[0])

    # 把数据划分为batch_size组
    batch_len = data_len // batch_size
    data_x = np.zeros([batch_size, batch_len], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_data[0][batch_len * i:batch_len * (i + 1)]
        data_y[i] = raw_data[1][batch_len * i:batch_len * (i + 1)]
    # 每一个epoch多少个batch
    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    # x shape is (batch_size,num_steps)
    # y shape is (batch_size,num_steps,4)
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


if __name__ == '__main__':
    read_data()
