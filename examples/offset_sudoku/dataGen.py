
import pickle
import numpy as np
import torch
from  torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler


def save_pickle(obj,filename):
    with open(filename+'.p', 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def load_pickle(file_to_load):
    with open(file_to_load, 'rb') as fp:
        labels = pickle.load(fp)
    return labels

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




image_file='image_dict_offset.p'
label_file='label_dict_offset.p'

#transform 

normalize = transforms.Normalize([.5],[.5])
preprocessing = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Grayscale(),
            transforms.Resize((250,250)), #resize images original was 250, large was 350, small was 175
            transforms.ToTensor(),
            normalize
            ])

#create dataset 
dataset=Sudoku_dataset(image_file,label_file,preprocessing,data_limit=5000)

#create train and test split 

X_unshuffled=dataset.indices

rs = ShuffleSplit(n_splits=1, test_size=.06, random_state=32)
rs.get_n_splits(X_unshuffled)


train_ind=[];val_ind=[];
for train_index,test_index in rs.split(X_unshuffled):
    train_ind.append(train_index);
    val_ind.append(test_index);


#randomly selected data indices
to_choose_from=np.array([ 26, 663, 279, 231, 589, 577, 687, 759, 177, 407, 752, 800, 294,
       624, 130, 736, 592, 363, 368,  88, 334, 635, 316,  91, 466, 298,
       203, 173, 359, 215,  37, 296, 205, 304, 711,   5, 134,  48, 307,
        51, 683,  81, 139, 529, 733,  85, 450, 158, 710,  30, 256, 610,
       138, 344, 200, 488,  68, 391, 254, 430,  35,  65, 504, 176, 305,
       206, 677, 727, 616, 284,   9, 218, 765,  17, 563, 401, 319, 153,
       507, 686, 225, 772, 666, 542, 819, 614, 293, 630, 741, 214, 642,
       605, 250, 498,  78, 705,  76, 627, 101,  13])


train_size=70 #maximum 100

pre_train_ind=np.arange(train_size)
train_ind=[to_choose_from[pre_train_ind]]
val_ind=[np.arange(825,925)] #test set is last 100 of dataset 


#create dataset 

batch_size=256

train_sampler = SubsetRandomSampler(train_ind[0].tolist())
valid_sampler = SubsetRandomSampler(val_ind[0].tolist())

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=64,
                                                sampler=valid_sampler)