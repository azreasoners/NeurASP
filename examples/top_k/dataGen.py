import re
import numpy as np
from numpy.random import permutation
import torch
from torch.autograd import Variable

item_number = 10

class KsData():
    def __init__(self, data_path='data/data.txt', item_number=10):
        np.random.seed(0)
        data = []
        labels = []
        self.item_number = item_number
        with open(data_path) as file:
            for line in file:
                tokens = re.compile('\[(.+)\]').split(line.strip())
                tokens = tokens[1].split('] [')
                data_1 = [float(i) for i in tokens[0].split(', ')]
                train_data = data_1
                train_label = [int(i) for i in tokens[1].split(', ')]
                data.append(train_data)
                labels.append(to_one_hot(train_label, self.item_number))

        perm = permutation(len(data))
        train_inds = perm[:int(len(data)*0.6)]
        
        valid_inds = perm[int(len(data)*0.6): int(len(data)*0.8)]
        test_inds = perm[int(len(data)*0.8):]
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.train_data = self.data[train_inds, :]
        self.valid_data = self.data[valid_inds, :]
        self.test_data = self.data[test_inds,:]
        self.train_labels = self.labels[train_inds,:]
        self.valid_labels = self.labels[valid_inds,:]
        self.test_labels = self.labels[test_inds,:]

        self.batch_ind = len(train_inds)
        self.batch_term = None

def to_one_hot(dense, n, inv=False):
    one_hot = np.zeros(n)
    one_hot[dense] = 1
    if inv:
        one_hot = (one_hot + 1) % 2
    return one_hot

def gen_weakconstraint():
    ks = KsData('data/data.txt', 10)
    train_value = ks.train_data[:,:ks.item_number]
    train_weight = ks.train_data[:,ks.item_number:]
    train_label = ks.train_labels
    
    valid_value = ks.valid_data[:,:ks.item_number]
    valid_weight = ks.valid_data[:,ks.item_number:]
    valid_label = ks.valid_labels
    
    test_value = ks.test_data[:,:ks.item_number]
    test_weight = ks.test_data[:,ks.item_number:]
    test_label = ks.test_labels
    begin1 = '#evidence'
    begin2 = ''
    value_begin = 'value('
    weight_begin = 'weight(' 
    with open('data/evidence_train.txt', 'w') as f:
        for index in range(len(ks.train_data)):
            f.write(begin1+'\n')
            f.write(begin2+'\n')
            for i in range(ks.item_number): 
                f.write(':~ in({}, k, true). [{}, {}]\n'.format(i, -int(train_value[index][i]), i))

    with open('data/evidence_valid.txt', 'w') as f:
        for index in range(len(ks.valid_data)):
            f.write(begin1+'\n')
            f.write(begin2+'\n')
            for i in range(ks.item_number): 
                f.write(':~ in({}, k, true). [{}, {}]\n'.format(i, -int(valid_value[index][i]), i))
    with open('data/evidence_test.txt', 'w') as f:
        for index in range(len(ks.test_data)):
            f.write(begin1+'\n')
            f.write(begin2+'\n')
            for i in range(ks.item_number): 
                f.write(':~ in({}, k, true). [{}, {}]\n'.format(i, -int(test_value[index][i]), i))

gen_weakconstraint()
dataset = KsData('data/data.txt',10)

# for training
dataList = []
obsList = []
for i, d in enumerate(dataset.train_data):
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    dataList.append({'k': d_tensor})
with open('data/evidence_train.txt', 'r') as f:
    obsList = f.read().strip().strip('#evidence').split('#evidence')

# for testing 
dataListTest = []
obsListTest = []
for d in dataset.test_data:
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    dataListTest.append({'k': d_tensor})
with open('data/evidence_test.txt', 'r') as f:
    obsListTest = f.read().strip().strip('#evidence').split('#evidence')
