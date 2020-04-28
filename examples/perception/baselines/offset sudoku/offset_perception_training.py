import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from sudoku_network import Sudoku_Net_Offset_bn,Sudoku_dataset,to_onehot
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt 
import torch.optim as optim
import time


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


breakpoint()

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

# =============================================================================
# Instantiate Network
# =============================================================================


model = Sudoku_Net_Offset_bn()
model.cuda()

# =============================================================================
# Network Parameters
# =============================================================================

learning_rate=.0001
epochs=2000

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

for epoch in range(epochs):  
    #n_of_batches=agent_data.len//batch_size
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
        
        print('Loss = {0}    Acc = {1} true_training_acc={2}'.format(loss,acc,true_acc))
        loss_hist.append(loss.data.item())
    
        opt.step()

    model.eval()
    acc=0
    correct=0.
    total=0.
    whole_acc_weighted=0.
    
    whole_val_denom=0.
    prompt_count=0
    
    if (epoch+1)%500==0 or epoch==epochs-1:   
        
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
            print('val_Loss = {0}    val_Acc = {1}  whole_board_val_acc ={2}'.format(round(float(loss),10),round(float(acc),10),round(float(true_acc),10)))

        whole_acc_weighted_hist.append(whole_acc_weighted/whole_val_denom); print('the length of whole_acc_weighted_hist is {0}'.format(len(whole_acc_weighted_hist)))
        print('\t\t\t\t\t\t\t batch_whole_board_whole_acc, is {0}'.format(whole_acc_weighted/whole_val_denom))
    
torch.save(model.state_dict(), 'model_{0}train'.format(train_size))




