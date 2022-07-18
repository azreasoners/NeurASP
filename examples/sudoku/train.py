import sys
sys.path.append('../../')
import time

import torch

from dataGen import dataList, obsList, train_loader, test_loader
from network import Sudoku_Net
from neurasp import NeurASP

startTime = time.time()

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
######################################

dprogram = '''
% neural rule
nn(identify(81, img), [empty,1,2,3,4,5,6,7,8,9]).

% we assign one number at each position (R,C)
a(R,C,N) :- identify(Pos, img, N), R=Pos/9, C=Pos\9, N!=empty.
{a(R,C,N): N=1..9}=1 :- identify(Pos, img, empty), R=Pos/9, C=Pos\9.

% it's a mistake if the same number shows 2 times in a row
:- a(R,C1,N), a(R,C2,N), C1!=C2.

% it's a mistake if the same number shows 2 times in a column
:- a(R1,C,N), a(R2,C,N), R1!=R2.

% it's a mistake if the same number shows 2 times in a 3*3 grid
:- a(R,C,N), a(R1,C1,N), R!=R1, C!=C1, ((R/3)*3 + C/3) = ((R1/3)*3 + C1/3).
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = Sudoku_Net()
nnMapping = {'identify': m}
optimizers = {'identify': torch.optim.Adam(m.parameters(), lr=0.0001)}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

########
# Set up the number of data to be used in training
########
try:
    numOfData = int(sys.argv[1])
except:
    numOfData = 29
dataList = dataList[:numOfData]
obsList = obsList[:numOfData]

########
# Start training from scratch and testing
########

saveModelPath = 'data/model_data{}.pt'.format(numOfData)

print('Use {} data to train the NN for 4000 epochs by NN only method (i.e., CrossEntropy loss)'.format(numOfData))
print(r'The identification accuracy of M_{identify} will also be printed out.\n')
for i in range(41):
    if i == 0:
        print('\nBefore Training ...')
    else:
        print('\nContinuously Training for 100 Epochs -- Round {} ...'.format(i))
        time1 = time.time()
        # here alpha=1 means rules are not used in training, in other words, it's usual NN training with cross entropy loss
        NeurASPobj.learn(dataList=dataList, obsList=obsList, alpha=1, epoch=100, lossFunc='cross', bar=True)
        time2 = time.time()
        print("--- train time: %s seconds ---" % (time2 - time1))        

    acc, singleAcc = NeurASPobj.testNN('identify', train_loader)
    print('Train Acc Using Pure NN (whole board): {:0.2f}%'.format(acc))
    print('Train Acc Using Pure NN (single cell): {:0.2f}%'.format(singleAcc))
    acc, singleAcc = NeurASPobj.testNN('identify', test_loader)
    print('Test Acc Using Pure NN (whole board): {:0.2f}%'.format(acc))
    print('Test Acc Using Pure NN (single cell): {:0.2f}%'.format(singleAcc))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )

# save the trained model
print('Storing the trained model into {}'.format(saveModelPath))
torch.save(m.state_dict(), saveModelPath)
