import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class Hasy_Apply(Dataset):

    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d1, d2, d3, o1, o2, o3, o4, r1, r2, c1, c2 = self.data[index] # self.dataset[o1][0] is of shape (1,32,32)
        return torch.cat((self.dataset[o1][0], self.dataset[o2][0], self.dataset[o3][0], self.dataset[o4][0]), 0).unsqueeze(1), d1, d2, d3, r1, r2, c1, c2

transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])
hasy_train_data = torchvision.datasets.ImageFolder(root = '../data/hasy/train', transform=transform)
hasy_test_data = torchvision.datasets.ImageFolder(root = '../data/hasy/test_apply2x2', transform=transform)
trainDataset = Hasy_Apply(hasy_train_data, '../data/apply2x2_train.txt')
# only randomly take 3000 data
np.random.seed(1) # fix the random seed for reproducibility
trainDataset = torch.utils.data.Subset(trainDataset, np.random.choice(len(trainDataset), 3000, replace=False))
testLoader = torch.utils.data.DataLoader(hasy_test_data, batch_size=1000, shuffle=True)

dataList = []
obsList = []
for images, d1, d2, d3, r1, r2, c1, c2 in trainDataset:
    dataList.append({'i': images})
    obsList.append(f':- not apply2x2({d1},{d2},{d3},{r1},{r2},{c1},{c2}).\ndigits({d1},{d2},{d3}).')
