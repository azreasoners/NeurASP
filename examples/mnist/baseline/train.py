import sys
sys.path.append("../../../")
import time

import torch

from dataGen import dataList, obsList, test_loader
from neurasp import NeurASP
from network import Net

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = '''
nn(addition(1,i), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]).
'''

########
# Define nnMapping and optimizers, initialze DeepLPMLN object
########

m = Net()
nnMapping = {'addition': m}
optimizers = {'addition': torch.optim.Adam(m.parameters(), lr=0.001)}

NeurASPobj = DeepLPMLN(dprogram, nnMapping, optimizers)

########
# Start training and testing
########

startTime = time.time()
for i in range(1):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1)
    time2 = time.time()
    acc, _ = NeurASPobj.testNN("m", test_loader)
    print('Test Acc: {:0.2f}%'.format(acc))
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )