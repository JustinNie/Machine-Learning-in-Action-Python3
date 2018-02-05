#classify digits based on svm
#Author: Justin Nie
#Date: 2018/2/2

from numpy import *
from svm import *

def image2vector(filename):
	'''convert an image to a vector'''
	vector = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		line_str = fr.readline()
		for j in range(32):
			vector[0, 32*i+j] = int(line_str[j])

	return vector

def load_images(filename):
	'''
		convert images to matrix
	'''
	from os import listdir
	labels = []
	file_list = listdir(filename)
	m = len(file_list)
	dataset = zeros((m, 1024))

	for i in range(m):
		filename_str = file_list[i]
		file_str = filename_str.split('.')[0]
		labels_str = int(file_str.split('_')[0])

		if labels_str == 9:
			labels.append(-1)
		else:
			labels.append(1)
		dataset[i, :] = image2vector('%s/%s' % (filename, filename_str))

	return dataset, labels 

def test_digits(k_tuple = ('rbf', 10)):
	'''

	'''
	
	dataset, labels = load_images('../Chapter2-KNN/trainingDigits')
	b, alphas = smo_simple_k(dataset, labels, 200, 0.0001, 10000, k_tuple)
	data_matrix = mat(dataset);		label_matrix = mat(labels).transpose()
	sv_index = nonzero(alphas.A > 0)[0]
	support_vectors = data_matrix[sv_index]
	labels_sv = label_matrix[sv_index]
	print("There are %d support vectors" % shape(support_vectors)[0])

	m, n = shape(data_matrix)
	error_count = 0
	for i in range(m):
		kernel_eval = kernel_trans(support_vectors, data_matrix[i, :], k_tuple)
		predict = kernel_eval.T * multiply(labels_sv, alphas[sv_index]) + b
		if sign(predict) != sign(labels[i]):
			error_count += 1
	print("the training error rate is: %f" % (float(error_count) / m))

	dataset, labels = load_images('../Chapter2-KNN/testDigits')
	data_matrix = mat(dataset);		label_matrix = mat(labels).transpose()
	m, n = shape(data_matrix)
	error_count = 0
	
	for i in range(m):
		kernel_eval = kernel_trans(support_vectors, data_matrix[i, :], k_tuple)
		predict = kernel_eval.T * multiply(labels_sv, alphas[sv_index]) + b
		if sign(predict) != sign(labels[i]):
			error_count += 1	
	print("the test error rate is: %f" % (float(error_count) / m))


test_digits(('rbf', 20))











































