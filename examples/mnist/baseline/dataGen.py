import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class MNIST_Addition(Dataset):

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
        i1, i2, l = self.data[index]
        return torch.cat((self.dataset[i1][0], self.dataset[i2][0]), 1), l

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])

train_dataset = MNIST_Addition(torchvision.datasets.MNIST(root='../../../data/', train=True, download=True, transform=transform), '../data/train_data.txt')
test_dataset = MNIST_Addition(torchvision.datasets.MNIST(root='../../../data/', train=False, download=True, transform=transform), '../data/test_data.txt')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=1)

dataList = []
obsList = []
for d, l in train_dataset:
    dataList.append({'i': d.unsqueeze(0)})
    obsList.append(':- not addition(i,0,{}).'.format(l))