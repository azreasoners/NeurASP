import sys
sys.path.append('../../')
import time

import torch

from dataGen import dataList, obsList, test_loader
from network import Net
from neurasp import NeurASP

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(A,0,N1), digit(B,0,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = Net()
nnMapping = {'digit':m}
optimizers = {'digit':torch.optim.Adam(m.parameters(), lr=0.001)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

########
# Start training and testing
########

startTime = time.time()
for i in range(1):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, alpha=0, smPickle='data/stableModels.pickle')
    time2 = time.time()
    acc, _ = NeurASPobj.testNN('digit', test_loader)
    print('Test Acc: {:0.2f}%'.format(acc))
    print('--- train time: %s seconds ---' % (time2 - time1))
    print('--- test time: %s seconds ---' % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )