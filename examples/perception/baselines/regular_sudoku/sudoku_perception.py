import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from sudoku_network import Sudoku_Net_2,Sudoku_dataset,to_onehot

from  torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import ShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt 
import torch.optim as optim


image_file='image_dict_reg_100.p'
label_file='label_dict_reg_100.p'

#transform 

normalize = transforms.Normalize([.5],[.5])
preprocessing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((250,250)), #resize images original was 250, large was 350, small was 175
            transforms.ToTensor(),
            normalize
            ])

#create dataset 
dataset=Sudoku_dataset(image_file,label_file,preprocessing,data_limit=5000)

#create train and test split 

X_unshuffled=dataset.indices

rs = ShuffleSplit(n_splits=1, test_size=.75, random_state=35)
rs.get_n_splits(X_unshuffled)


train_ind=[];val_ind=[];
for train_index,test_index in rs.split(X_unshuffled):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_ind.append(train_index);
    val_ind.append(test_index);


to_choose_from=np.array([59, 45,  9, 79, 19,  3, 21, 61, 76, 94, 91, 88, 44, 97, 57, 72, 56,
        93, 11, 64, 63, 33, 55, 15, 73])

train_size=25
pre_train_ind=np.arange(train_size)
train_ind=[to_choose_from[pre_train_ind]]


val_ind=[np.arange(825,925)]
val_ind=[np.array([0,1,2,4,5,6,7,8,10,12,13,14,16,17,18,20,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,
 38,39,40,41,42,43,46,47,48,49,50,51,52,53,54,58,60,62,65,66,67,68,69,70,71,74,75,77,
 78,80,81,82,83,84,85,86,87,89,90,92,95,96,98,99])]

#create dataset 

batch_size=256


train_sampler = SubsetRandomSampler(train_ind[0].tolist())
valid_sampler = SubsetRandomSampler(val_ind[0].tolist())

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=75,
                                                sampler=valid_sampler)

# =============================================================================
# Instantiate Network
# =============================================================================

model = Sudoku_Net_2()
model.cuda()


# =============================================================================
# Network Parameters
# =============================================================================

learning_rate=.0001
epochs=10000

opt=optim.Adam(model.parameters(),lr=learning_rate)
criterion=torch.nn.BCELoss()

opt.zero_grad()
acc_hist_train=[]
acc_hist_val=[]
loss_hist=[]
sample_hist=[]

whole_board_acc=[]
whole_acc_weighted_hist=[]
# =============================================================================
# Training
# =============================================================================

val_mod=1000

for epoch in range(epochs):  

    print('epoch {0} out of {1}'.format(epoch+1,epochs))
    model.train()
    acc=0
    correct=0.
    total=0.
    for i_batch,batch in enumerate(train_loader):

        opt.zero_grad()
        x=batch[0].cuda()
        
        labels=batch[1].float()
        labels_acc=labels.view(-1,81)
        labels=to_onehot(labels,len(labels)).cuda()
        
        outputs_acc=model(x)
        outputs=outputs_acc.view(len(labels),-1)
        outputs_acc=outputs_acc.max(2)[1].float() # take max over the probabilities
        loss = criterion(outputs,labels)
        loss.backward()
        
        labels_acc=labels_acc.cuda()
        
        acc=float((outputs_acc==labels_acc).sum())/outputs_acc.numel()
        acc_hist_train.append(acc)
        true_acc=0.
        
        for l,o in zip(labels_acc,outputs_acc):
            if (l.tolist()==o.tolist()):
                true_acc+=1
        true_acc/=len(labels_acc)
        
        print('Loss = {0}    grid_cell_acc = {1} whole_board_acc={2}'.format(loss,acc,true_acc))
        loss_hist.append(loss.data.item())
    
        opt.step()

    
    model.eval()
    acc=0
    correct=0.
    total=0.
    whole_acc_weighted=0.
    
    whole_val_denom=0.
    
    if (epoch+1)%val_mod==0:
    
        for i_batch,batch in enumerate(validation_loader):
            
            opt.zero_grad()
            x=batch[0].cuda()
            labels=batch[1].float()
            labels_acc=labels.view(-1,81)
            labels=to_onehot(labels,len(labels)).cuda()
            
            outputs_acc=model(x)
            outputs=outputs_acc.view(len(labels),-1)
            outputs_acc=outputs_acc.max(2)[1].float()
            
            loss = criterion(outputs,labels)
            
            labels_acc=labels_acc.cuda()
            acc=float((outputs_acc==labels_acc).sum())/outputs_acc.numel()
            
            acc_hist_val.append(acc)
            true_acc=0.
            whole_val_denom+=len(batch[0])
            
            for l,o in zip(labels_acc,outputs_acc):
                if (l.tolist()==o.tolist()):
                    true_acc+=1

            true_acc/=len(labels_acc)
            whole_acc_weighted+=true_acc*len(batch[0])
            whole_board_acc.append(true_acc)
            loss = criterion(outputs,labels)
            print('val_Loss = {0}    grid_cell_acc = {1}  whole_board_acc ={2}'.format(round(float(loss),10),round(float(acc),10),round(float(true_acc),10)))

        whole_acc_weighted_hist.append(whole_acc_weighted/whole_val_denom);
        print('\t\t\t\t\t\t\t whole_board_whole_validation, is {0}'.format(whole_acc_weighted/whole_val_denom))
    
torch.save(model.state_dict(), 'model_reg{0}train'.format(train_size))







