import sys
sys.path.append('../../')

import torch

from dataGen import termPath2dataList
from network import Net
from neurasp import NeurASP

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = r'''
nn(label(1,I,B), ["car", "cat", "person", "truck", "other"]) :- box(I,B,X1,Y1,X2,Y2).
'''

aspProgram = r'''
% define smaller/2 between labels
smaller("cat", "person").
smaller("person", "car").
smaller("person", "truck").
smaller(X,Y) :- smaller(X,Z), smaller(Z,Y).

% define smaller/3 between objects in bounding boxes
smaller(I,B1,B2) :- not -smaller(I,B1,B2), label(0,I,B1,L1), label(0,I,B2,L2), smaller(L1,L2).
-smaller(I,B2,B1) :- box(I,B1,X1,Y1,X2,Y2), box(I,B2,X1',Y1',X2',Y2'), Y2>=Y2', |X1-X2|*|Y1-Y2| < |X1'-X2'|*|Y1'-Y2'|.
smaller(I,B1,B2) :- -smaller(I,B2,B1).
toy(I,B1) :- label(0,I,B1,L1), label(0,I,B2,L2), smaller(I,B1,B2), smaller(L2,L1).
'''

########
# Define nnMapping
########

m = Net()
nnMapping = {'label': m}

########
# Construct a list of facts and a list of dataDic, where each dataDic maps terms to tensors
########

# set the term and the path to the image files represetned by this term
termPath = 'img ./data/'
# set the size of the reshaped image
img_size = 416
# set the set of classes that we consider
domain = ["car", "cat", "person", "truck", "other"]

factsList, dataList = termPath2dataList(termPath, img_size, domain)

########
# Start inference for each image
########

for idx, facts in enumerate(factsList):
    # Initialize NeurASP object
    NeurASPobj = NeurASP(dprogram + facts, nnMapping, optimizers=None)
    # Find the most probable stable model
    models = NeurASPobj.infer(dataDic=dataList[idx], obs='', mvpp=aspProgram + facts)
    print('\nInfernece Result on Data {}:'.format(idx+1))
    print(models[0])
