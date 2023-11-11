import numpy as np
from numpy.random import permutation
import torch
from torch.autograd import Variable

# Define the class for the dataset
class GridData():
    def __init__(self, data_path):
        np.random.seed(0)
        data = []
        labels = []
        with open(data_path) as file:
            for line in file:
                tokens = line.strip().split(',')
                if(tokens[0] != ''):
                    removed = [int(x) for x in tokens[0].split('-')]
                else:
                    removed = []

                inp = [int(x) for x in tokens[1].split('-')]
                paths = tokens[2:]
                data.append(np.concatenate((self.to_one_hot(removed, 24, inv=True), self.to_one_hot(inp, 16))))
                path = [int(x) for x in paths[0].split('-')]
                labels.append(self.to_one_hot(path, 24))

        # We're going to split 60/20/20 train/test/validation
        perm = permutation(len(data))
        train_inds = perm[:int(len(data)*0.6)]
        valid_inds = perm[int(len(data)*0.6):int(len(data)*0.8)]
        test_inds = perm[int(len(data)*0.8):]
        data = np.array(data)
        labels = np.array(labels)

        np.random.seed()

        self.dic = {}
        self.dic['train'] = data[train_inds, :]
        self.dic['test'] = data[test_inds, :]
        self.dic['valid'] = data[valid_inds, :]
        self.dic['train_label'] = labels[train_inds, :]
        self.dic['test_label'] = labels[test_inds, :]
        self.dic['valid_label'] = labels[valid_inds, :]

    @staticmethod
    def to_one_hot(dense, n, inv=False):
        one_hot = np.zeros(n)
        one_hot[dense] = 1
        if inv:
            one_hot = (one_hot + 1) % 2
        return one_hot

# Define the function that turns dataset files (under text form) into dataset (dictionary) and evidence files
def generateDataset(inPath, outPath):
    grid_data = GridData(inPath)
    names = ['train', 'test', 'valid']
    for name in names:
        fname = outPath+name+'.txt'
        with open(fname, 'w') as f:
            for data in grid_data.dic[name].tolist():
                removed = data[:24]
                startEnd = data[24:]
                removed = [i for i, x in enumerate(removed) if x == 0]
                startEnd = [i for i, x in enumerate(startEnd) if x == 1]
                evidence = ':- mistake.\n'
                for edge in removed:
                    evidence += 'removed({}).\n'.format(edge)
                for node in startEnd:
                    evidence += 'sp(external, {}).\n'.format(node)
                f.write(evidence)
                f.write('#evidence\n')
    return grid_data.dic

# construct the dataset (a dictionary)
dataset = generateDataset('data/shortestPath.data', 'data/shorteatPath_')

# for training
training_just_with_labels = False
obsList = []
with open('data/shorteatPath_train.txt', 'r') as f:
    obsList = f.read().strip().strip('#evidence').split('#evidence')
dataList = []
for data, label in zip(dataset['train'], dataset['train_label']):
    if training_just_with_labels:
        dataList.append({'g': (Variable(torch.from_numpy(data).float(), requires_grad=False), {'sp': Variable(torch.from_numpy(label).float().view(-1, 1), requires_grad=False)})})
    else:
        dataList.append({'g': Variable(torch.from_numpy(data).float(), requires_grad=False)})

# for testing
obsListTest = []
with open('data/shorteatPath_test.txt', 'r') as f:
    obsListTest = f.read().strip().strip('#evidence').split('#evidence')
dataListTest = []
for data in dataset['test']:
    dataListTest.append({'g': Variable(torch.from_numpy(data).float(), requires_grad=False)})
