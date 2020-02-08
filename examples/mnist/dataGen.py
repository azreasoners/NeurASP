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
        return self.dataset[i1][0], self.dataset[i2][0], l

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])

train_dataset = MNIST_Addition(torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform), 'data/train_data.txt')
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, transform=transform), batch_size=1000, shuffle=True)
dataList = []
obsList = []
for i1, i2, l in train_dataset:
    dataList.append({'i1': i1.unsqueeze(0), 'i2': i2.unsqueeze(0)})
    obsList.append(':- not addition(i1, i2, {}).'.format(l))