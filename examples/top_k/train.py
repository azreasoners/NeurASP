import sys
sys.path.append('../../')

import numpy as np
import random
import time
import torch
from torch.autograd import Variable

from dataGen import KsData
from network import FC
from neurasp import NeurASP

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram='''
% define maxweight k 
#const k = 7.

nn(in(10, k), [true, false]).

% we make a mistake if the total weight of the chosen items exceeds maxweight 
:- #sum{1, I : in(k,I,true)} > k.
'''

dprogram_test='''
% define maxweight k 
#const k = 7.

% we make a mistake if the total weight of the chosen items exceeds maxweight 
:- #sum{1, I : in(k,I,true)} > k.
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = FC(10, *[50, 50, 50, 50, 50], 10)
nnMapping = {'in': m}
optimizer = {'in':torch.optim.Adam(m.parameters(), lr=0.001)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizer)

dataset = KsData('data/data.txt',10)
dataList = []
obsList = []
for i, d in enumerate(dataset.train_data):
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    dataList.append({'k': d_tensor})
with open('data/evidence_train.txt', 'r') as f:
    obsList = f.read().strip().strip('#evidence').split('#evidence')

# testing 
testData = []
testObsLost = []
for d in dataset.test_data:
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    testData.append({'k': d_tensor})
with open('data/evidence_test.txt', 'r') as f:
    testObsLost = f.read().strip().strip('#evidence').split('#evidence')


########
# Start training and testing
########

startTime = time.time()
for i in range(200):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, alpha=0, opt=True, smPickle='data/stableModels.pickle')
    time2 = time.time()
    NeurASPobj.testConstraint(testData, testObsLost,[dprogram_test])
    print('--- train time: %s seconds ---' % (time2 - time1))
    print('--- test time: %s seconds ---' % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )
