#Singular Value Decomposition
#Author: Justin Nie
#Date: 2018/2/15

from numpy import *

def load_simdata():
	'''
		return a list of two dimensions
	'''
	dataset = [[1, 1, 1, 0, 0],
			   [2, 2, 2, 0, 0],
			   [1, 1, 1, 0, 0],
			   [5, 5, 5, 0, 0],
			   [1, 1, 0, 2, 2],
			   [0, 0, 0, 3, 3],
			   [0, 0, 0, 1, 1],
			  ]
	return dataset

def load_simdata2():
	dataset =  [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
				[0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
				[0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
				[3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
				[5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
				[0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
				[4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
				[0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
				[0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
				[0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
				[1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0], 
			   ]
	return dataset

def load_dataset(filename):
	'''
		load the dataset
	'''
	fr = open(filename)
	dataset = []
	for line in fr.readlines():
		line_list = []
		for index in range(len(line) - 1):
			line_list.append(int(line[index]))
		dataset.append(line_list)

	return dataset


def svd(dataset, threshold=0.9):
	'''
		decomposite the dataset
		the main singular values are more than 90%
	'''
	u, sigma, vt = linalg.svd(dataset)
	sum_value = sum(abs(sigma))
	add_value = 0;	index=0
	for singular_value in sigma:
		add_value += abs(singular_value)
		if add_value/sum_value >= threshold:
			print("Percent: ", add_value/sum_value)
			break
		index += 1

	i=0
	value_list = [sigma[i] for i in range(index+1)]
	sigma_mat = diag(value_list)

	new_data_mat = mat(u[:, :index+1]) * mat(sigma_mat) * mat(vt[:index+1, :])

	return u, mat(sigma_mat), vt, new_data_mat

def euclidian_similarity(vector_a, vector_b):
	'''
		Euclidian similarity
		a and b are both column vectors
	'''
	return 1.0 / (1.0 + linalg.norm(vector_a - vector_b))

def pearson_similarity(vector_a, vector_b):
	'''
		Pearson similarity
		a and b are both column vectors

	'''
	if len(vector_a) < 3:
		return 1.0
	return 0.5 + 0.5*corrcoef(vector_a, vector_b, rowvar=0)[0][1]

def cosine_similarity(vector_a, vector_b):
	'''
		cosine similarity
		a and b are both column vectors

	'''
	numerator = float(vector_a.T * vector_b)
	denominator = linalg.norm(vector_a) * linalg.norm(vector_b)

	return 0.5 + 0.5 * (numerator / denominator)
