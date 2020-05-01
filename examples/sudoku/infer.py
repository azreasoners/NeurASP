import sys
sys.path.append('../../')

import torch

from dataGen import loadImage
from network import Sudoku_Net
from neurasp import NeurASP

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
######################################

dprogram = '''
% neural rule
nn(identify(81, img), [empty,1,2,3,4,5,6,7,8,9]).
'''

rules = {}

rules['normal'] = '''
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

rules['anti-knight'] = rules['normal'] + '''
:- a(R1,C1,N), a(R2,C2,N), |R1-R2|+|C1-C2|=3.
'''

rules['Sudoku-X'] = rules['normal'] + '''
:- a(R1,C1,N), a(R2,C2,N), R1=C1, R2=C2, R1!=R2.
:- a(R1,C1,N), a(R2,C2,N), R1+C1=8, R2+C2=8, R1!=R2.
'''

rules['offset'] = rules['normal'] + '''
:- a(R1,C1,N), a(R2,C2,N), R1\3 = R2\3, C1\3 = C2\3, R1 != R2, C1 != C2.
'''

########
# Define nnMapping and optimizers, initialze DeepLPMLN object
########

m = Sudoku_Net()
nnMapping = {'identify': m}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers=None)

########
# Obtain the type of Sudoku and the path to the image from command line arguments
########
try:
    sudokuType = sys.argv[1]
    imagePath = sys.argv[2]
except:
    print('Error: please make sure your command follows the format: python infer.py SudokuType IMAGE')
    print('SudokuType should be one of {normal, anti-knight, Sudoku-X, offset}')
    print('\ne.g. python infer.py normal data/sudoku.png')
    sys.exit()

assert sudokuType in rules, r'Error: the given Sudoku type should be in the set {normal, anti-knight, Sudoku-X, offset}'

try:
    image = loadImage(imagePath)
except:
    print('Error: cannot load the image')
    sys.exit()

########
# Load pre-trained model
########

numOfData = 25
saveModelPath = 'data/model_data25.pt'.format(numOfData)
print('\nLoad the model trained with {} instances of normal Sudoku puzzles'.format(numOfData))
m.load_state_dict(torch.load('data/model_data{}.pt'.format(numOfData), map_location='cpu'))

########
# Start infering on the given image
########

dataDic = {'img': image}
models = NeurASPobj.infer(dataDic=dataDic, mvpp=rules[sudokuType])
print('\nInference Result:\n', models[0])
