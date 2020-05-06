import sys
sys.path.append('../../')

import torch

from dataGen import loadImage
from network import Sudoku_Net_Offset_bn
from neurasp import NeurASP

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
######################################

dprogram = '''
% neural rule
nn(identify(81, img), [empty,1,2,3,4,5,6,7,8,9]).
'''

aspProgram = '''
% we assign one number at each position (R,C)
a(R,C,N) :- identify(Pos, img, N), R=Pos/9, C=Pos\9, N!=empty.
{a(R,C,N): N=1..9}=1 :- identify(Pos, img, empty), R=Pos/9, C=Pos\9.

% it's a mistake if the same number shows 2 times in a row
:- a(R,C1,N), a(R,C2,N), C1!=C2.

% it's a mistake if the same number shows 2 times in a column
:- a(R1,C,N), a(R2,C,N), R1!=R2.

% it's a mistake if the same number shows 2 times in a 3*3 grid
:- a(R,C,N), a(R1,C1,N), R!=R1, C!=C1, ((R/3)*3 + C/3) = ((R1/3)*3 + C1/3).

% rule for offset sudoku
:- a(R1,C1,N), a(R2,C2,N), R1\3 = R2\3, C1\3 = C2\3, R1 != R2, C1 != C2.
'''

########
# Define nnMapping and initialze NeurASP object
########

m = Sudoku_Net_Offset_bn()
nnMapping = {'identify': m}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers=None)

########
# Obtain the path to the image from command line arguments
########

try:
    imagePath = sys.argv[1]
except:
    print('Error: please make sure your command follows the format: python infer.py IMAGE')
    print('e.g. python infer.py data/offset_sudoku.png')
    sys.exit()

try:
    image = loadImage(imagePath)
except:
    print('Error: cannot load the image')
    sys.exit()

########
# Load pre-trained model
########

numOfData = 70
saveModelPath = 'data/model_data{}.pt'.format(numOfData)
print('\nLoad the model trained with {} instances of normal Sudoku puzzles'.format(numOfData))
m.load_state_dict(torch.load('data/model_data{}.pt'.format(numOfData), map_location='cpu'))

########
# Start infering on the given image
########

dataDic = {'img': image}
models = NeurASPobj.infer(dataDic=dataDic, mvpp=aspProgram)
print('\nInference Result:\n', models[0])
