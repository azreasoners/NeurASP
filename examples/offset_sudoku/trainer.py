
from dataGen import to_onehot


def Train_Test(model, train_loader, validation_loader, opt, criterion, epochs):
    print("Training...")
    opt.zero_grad()
    acc_hist_train=[]
    acc_hist_val=[]
    loss_hist=[]
    sample_hist=[]
    
    whole_board_acc=[]
    whole_acc_weighted_hist=[]
    
    
    for epoch in range(epochs):
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
            
            if (epoch+1)%100==0:
                print('epoch {0} out of {1}'.format(epoch+1,epochs))
                print('Training set (single batch): Loss = {0}    grid_cell_acc = {1} whole_board_acc={2}'.format(loss,acc,true_acc))
            loss_hist.append(loss.data.item())
        
            opt.step()
        
        model.eval()
        acc=0
        correct=0.
        total=0.
        whole_acc_weighted=0.
        whole_val_denom=0.
        
        if (epoch+1)%100==0:
        
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
        
                whole_acc_weighted_hist.append(whole_acc_weighted/whole_val_denom); 

            print('validation set             : Loss = {0}     grid_cell_acc = {1} whole_board_acc, is {2}'.format(loss, acc, whole_acc_weighted/whole_val_denom))
    
    return model

def Test(model, train_loader, validation_loader, opt, criterion, epochs):
    print("Testing...")
    opt.zero_grad()
    acc_hist_train=[]
    acc_hist_val=[]
    loss_hist=[]
    sample_hist=[]
    
    whole_board_acc=[]
    whole_acc_weighted_hist=[]
    
    
    for epoch in range(epochs):
        
        model.eval()
        acc=0
        correct=0.
        total=0.
        whole_acc_weighted=0.
        whole_val_denom=0.
        
        if (epoch+1)%100==0:
        
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
        
                whole_acc_weighted_hist.append(whole_acc_weighted/whole_val_denom); 

            print('validation set             : Loss = {0}     grid_cell_acc = {1} whole_board_acc, is {2}'.format(loss, acc, whole_acc_weighted/whole_val_denom))