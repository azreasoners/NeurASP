import numpy as np 
import random
from numpy.random import permutation
import sys

edges_number = 24
grid_size = 4

def to_one_hot(dense, n, inv=False):
	one_hot = np.zeros(n)
	one_hot[dense] = 1
	if inv:
		one_hot = (one_hot + 1) % 2
	return one_hot

class GridProbData():
	def __init__(self, file):
		with open(file) as f:
			d = f.readlines()
			input_data = []
			output_data = []
			for data in d:
				data = data.strip().split(' ')
				two_point = [int(float(x)) for x in data[:2]]
				two_point = to_one_hot(two_point, grid_size*grid_size)
				prob = [float(x) for x in data[2:2+edges_number]]
				labels = [float(x) for x in data[2+edges_number:]]
				
				input_data.append(np.concatenate((np.array(prob),two_point)))
				output_data.append(labels)
		length = len(input_data)
		input_data = np.array(input_data)
		output_data = np.array(output_data)
		print('there are {} data in total, 60% training data, 20% validation data, 20% testing data!'.format(length))
		self.train_data = input_data[:int(length*0.6)]
		self.train_labels = output_data[:int(length*0.6)]
		self.valid_data = input_data[int(length*0.6):int(length*0.8)]
		self.valid_labels = output_data[int(length*0.6):int(length*0.8)]
		self.test_data = input_data[int(length*0.8):]
		self.test_labels = output_data[int(length*0.8):]

def generate_weak_con_evidence(file):
	gd = GridProbData(file)
	train_data = gd.train_data
	labels = gd.train_labels
	length = len(train_data)
	end = '#evidence'
	begin = ''
	external_begin = 'sp(external, '
	with open('data/evidence_train.txt','w') as f:
		for l in range(length):
			f.write(begin+'\n')
			initial_nodes = np.where(train_data[l][edges_number:] == 1)[0]
			probs_edges = train_data[l][:edges_number]
			for n in initial_nodes:
				f.write(external_begin+str(n)+').\n')
			s =''
			for i, p in enumerate(train_data[l][:edges_number]):
			    if p == 0.512:
			        f.write(':~ sp(g, {}, true). [{},{}]\n'.format(i, 6, i))
			    else:
			        f.write(':~ sp(g, {}, true). [{},{}]\n'.format(i, 2, i))
			f.write(end+'\n')
	test_data = gd.test_data
	length = len(test_data)
	end = '#evidence'
	begin = ''
	external_begin = 'sp(external, '
	with open('data/evidence_test.txt','w') as f:
		for l in range(length):
			f.write(begin+'\n')
			initial_nodes = np.where(test_data[l][edges_number:] == 1)[0]
			probs_edges = test_data[l][:edges_number]
			for n in initial_nodes:
				f.write(external_begin+str(n)+').\n')
			s =''
			for i, p in enumerate(test_data[l][:edges_number]):
			    if p == 0.512:
			        f.write(':~ sp(g, {}, true). [{},{}]\n'.format(i, 6, i))
			    else:
			        f.write(':~ sp(g, {}, true). [{},{}]\n'.format(i, 2, i))
			f.write(end+'\n')
	print('done!')

generate_weak_con_evidence('data/data.txt')