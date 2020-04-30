import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import torch
from  torch.utils.data import Dataset, DataLoader
import pickle 



    
def save_pickle(obj,filename):
    with open(filename+'.p', 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def load_pickle(file_to_load):
    with open(file_to_load, 'rb') as fp:
        labels = pickle.load(fp)
    return labels


class Sudoku_Net_Offset_bn(nn.Module):
    #add relu after 1x1conv add FC layer, dropout, adaptive pooling
    def __init__(self):
        super(Sudoku_Net_Offset_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32,kernel_size=4,stride=2)
        self.conv1_bn=nn.BatchNorm2d(32)
        self.dropout1=nn.Dropout(p=.25)
        self.conv2 = nn.Conv2d(32, 64,kernel_size=3,stride=2)
        self.conv2_bn=nn.BatchNorm2d(64)
        self.dropout2=nn.Dropout(p=.25)
        self.conv3 = nn.Conv2d(64, 128,kernel_size=3,stride=2)
        self.conv3_bn=nn.BatchNorm2d(128)
        self.dropout3=nn.Dropout(p=.25)
        self.conv4 = nn.Conv2d(128, 256,kernel_size=2,stride=1)
        self.conv4_bn=nn.BatchNorm2d(256)
        self.dropout4=nn.Dropout(p=.25)
        self.conv5 = nn.Conv2d(256, 512,kernel_size=2,stride=1)
        self.conv5_bn=nn.BatchNorm2d(512)
        self.dropout5=nn.Dropout(p=.25)
        
        self.maxpool=nn.MaxPool2d(3)
        
        self.conv1x1_1=nn.Conv2d(in_channels=512,out_channels=10,kernel_size=1)
        
        
    def forward(self, x):
        x = self.dropout1(self.conv1_bn(self.conv1(x)))
        x = F.relu(x)
        x = self.dropout2(self.conv2_bn(self.conv2(x)))
        x = F.relu(x)
        x = self.dropout3(self.conv3_bn(self.conv3(x)))
        x = F.relu(x)
        x = self.dropout4(self.conv4_bn(self.conv4(x)))
        x = F.relu(x)
        x = self.dropout5(self.conv5_bn(self.conv5(x)))
        x = F.relu(x)
        
        x= self.maxpool(x)
        x=self.conv1x1_1(x)
        x=nn.Softmax(1)(x)
        batch_size=len(x)
        x=x.permute(0,2,3,1).contiguous().view(batch_size,810)
        
        x=x.view(batch_size,81,10)
        return x







class Sudoku_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_path,label_path,transform=None,data_limit=99999):

        self.transform = transform
        self.image_dict=load_pickle(image_path)
        self.label_dict=load_pickle(label_path)

        
        keys_to_delete=[]
        keep_first_n=data_limit
        for i,key in enumerate(self.label_dict):
            if i>keep_first_n:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.label_dict[key]

        self.indices=np.arange(len(self.label_dict))
        self.idx_to_filename = {key:value for key,value in enumerate(self.label_dict.keys())}
        
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        filename=self.idx_to_filename[idx]
        
        x=self.image_dict[filename]
        
        x=torch.from_numpy(x).permute(2,0,1).float()
        
        y=self.label_dict[filename]

        if self.transform:
            x = self.transform(x)

        return x,y


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

