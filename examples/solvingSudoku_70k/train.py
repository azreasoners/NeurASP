import sys
sys.path.append('../../')
import time

import torch

from dataGen import dataList, obsList, train_loader, test_loader
from neurasp import NeurASP
from network import Sudoku_Net

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = '''
% neural rule
nn(predict(81, config), [1,2,3,4,5,6,7,8,9]).

% we assign one number at each position (R,C)
a(R,C,N) :- predict(config, Pos, N), R=Pos/9, C=Pos\9.

% it's a mistake if the same number shows 2 times in a row
:- a(R,C1,N), a(R,C2,N), C1!=C2.

% it's a mistake if the same number shows 2 times in a column
:- a(R1,C,N), a(R2,C,N), R1!=R2.

% it's a mistake if the same number shows 2 times in a 3*3 grid
:- a(R,C,N), a(R1,C1,N), R!=R1, C!=C1, ((R/3)*3 + C/3) = ((R1/3)*3 + C1/3).
'''

########
# Define nnMapping and optimizers, initialze DeepLPMLN object
########

m = Sudoku_Net()
nnMapping = {'predict': m}
optimizers = {'predict': torch.optim.Adam(m.parameters(), lr=0.0001)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizers, gpu=False)

########
# Start training and testing
########

# train a model using DeepLPMLN starting from a pre-trained model 
# m.load_state_dict(torch.load('data/model0107_alpha0_data70k_epoch22', map_location='cpu'))

# train a model from scratch using NN only
startTime = time.time()
print('Initial test accuracy (whole board): {:0.2f}%\nInitial test accuracy (single cell): {:0.2f}%'.format(*NeurASPobj.testNN('predict', test_loader)))

for i in range(100):
    print('Training for Epoch {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, alpha=0, epoch=1, method='exact', smPickle='data/stableModels.pickle', batchSize=1)
    time2 = time.time()
    acc, singleAcc = NeurASPobj.testNN('predict', train_loader)
    print('Train Acc (whole board): {:0.2f}%'.format(acc))
    print('Train Acc (single cell): {:0.2f}%'.format(singleAcc))
    acc, singleAcc = NeurASPobj.testNN('predict', test_loader)
    print('Test Acc (whole board): {:0.2f}%'.format(acc))
    print('Test Acc (single cell): {:0.2f}%'.format(singleAcc))
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )
    
    saveModelPath = 'data/model_data70k_alpha0_batchSize1_epoch{}'.format(i+1)
    print('Storing the trained model into {}'.format(saveModelPath))
    torch.save(m.state_dict(), saveModelPath)