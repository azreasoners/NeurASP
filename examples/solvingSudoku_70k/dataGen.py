import pickle
import numpy as np
from sklearn.model_selection import ShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

def save_pickle(obj,filename):
    with open(filename+'.p', 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def load_pickle(file_to_load):
    with open(file_to_load, 'rb') as fp:
        labels = pickle.load(fp)
    return labels

class Sudoku_dataset(Dataset):
    def __init__(self, input_path,label_path,transform=None,data_limit=99999999):
        self.transform = transform
        self.input_dict=load_pickle(input_path)
        self.label_dict=load_pickle(label_path)
        keys_to_delete=[]
        keep_first_n=data_limit
        for i,key in enumerate(self.label_dict):
            if i>=keep_first_n:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.label_dict[key]
        self.indices=np.arange(len(self.label_dict))
        self.idx_to_filename = {key:value for key,value in enumerate(self.label_dict.keys())}
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        filename=self.idx_to_filename[idx]
        x=self.input_dict[filename]
        x=torch.from_numpy(x)
        x=x.unsqueeze(0).float()
        y=self.label_dict[filename]
        y=torch.from_numpy(y)
        y=y-1
        return x,y.reshape(81)

def to_onehot(y,batch_size):
    ''' creates a one hot vector for the labels. y_onehot will be of shape (batch_size, 810)'''
    nb_digits=10
    one_hot_labels=torch.zeros(batch_size,810)
    for i,v in enumerate(y):
        y_onehot = torch.FloatTensor(81, nb_digits)
        y_onehot.zero_()
        y_onehot.scatter_(1, v.view(-1,1).long(), 1)
        one_hot_labels[i]=y_onehot.view(-1)
    return one_hot_labels


# initialze the dataset
input_file = 'data/easy_130k_given.p'
solved_file = 'data/easy_130k_solved.p'
dataset = Sudoku_dataset(input_file, solved_file, data_limit=70000)

X_unshuffled = dataset.indices
rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=32)
rs.get_n_splits(X_unshuffled)

train_ind=[]
val_ind=[]
for train_index,test_index in rs.split(X_unshuffled):
    train_ind.append(train_index)
    val_ind.append(test_index)

# create dataset 
batch_size=1

train_sampler = SubsetRandomSampler(train_ind[0].tolist())
test_sampler = SubsetRandomSampler(val_ind[0].tolist())

train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)

# construct dataList and obsList for training
dataList = []
obsList = []

for X, y in train_loader:
    y = y.view(-1).long()
    tmp = (X, {'sol': y})
    dataList.append({'config': tmp})
    obs = ''
    for pos, value in enumerate(X.view(-1).tolist()):
        if value != 0:
            obs += ':- not sol({}, config, {}).\n'.format(pos, int(value))
    obsList.append(obs)

train_loader = DataLoader(dataset, batch_size=230, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=230, sampler=test_sampler)