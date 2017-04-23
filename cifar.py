from __future__ import print_function
import _pickle as cPickle
import numpy as np


def load_cifar_file(path='../input/data_batch_1'):
    data_file = open(path, 'rb')
    data_dict = cPickle.load(data_file, encoding='latin-1')

    images = data_dict['data'].reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])
    if 'fine_labels' in data_dict:
      labels = np.array(data_dict['fine_labels'])
    else:
      labels = np.array(data_dict['labels'])
    return images, labels

def load_cifar(path='./input'):
  images = []; labels= []
  for batch_name in ('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch'):
    i, l = load_cifar_file(path + '/' + batch_name)
    images.append(i); labels.append(l)
  print(images, labels)


if __name__ == '__main__':
    load_cifar('./input')