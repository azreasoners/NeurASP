import sys
sys.path.append('../../')

import torch

from dataGen import test_loader
from network import Net
from neurasp import NeurASP

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

########
# Define nnMapping and initialze NeurASP object
########

m = Net()
nnMapping = {'digit': m}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers=None)

########
# Load pretrained model
########

saveModelPath = 'data/model.pt'
m.load_state_dict(torch.load(saveModelPath, map_location='cpu'))

########
# Start testing
########
acc, _ = NeurASPobj.testNN('digit', test_loader)
print('Test Acc on {} test data: {:0.2f}%'.format(len(test_loader)*test_loader.batch_size, acc))
