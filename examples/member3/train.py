import os
import sys
sys.path.append('../../')
import time

import torch

from dataGen import dataList, obsList, testLoader
from network import Net
from neurasp import NeurASP

startTime = time.time()

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = '''
nn(digit(3,i), [0,1,2,3,4,5,6,7,8,9]).
member(D,0) :- digit(0,i,N1), digit(1,i,N2), digit(2,i,N3), 
               check(D), D!=N1, D!=N2, D!=N3.
member(D,1) :- check(D), not member(D,0).
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = Net()
nnMapping = {'digit': m}
optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

########
# Start training and testing
########

# remove the saved models to fairly check total training time
try:
    os.remove('../data/member3_models.pickle')
except OSError:
    pass

for i in range(3):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, smPickle='../data/member3_models.pickle', bar=True)
    time2 = time.time()
print('--- total time for training: %s seconds ---' % int((time.time() - startTime)) )
acc, _ = NeurASPobj.testNN('digit', testLoader)
print('Test Acc: {:0.2f}%'.format(acc))
print('--- total time from beginning: %s seconds ---' % int((time.time() - startTime)) )