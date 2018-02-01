#Simple KNN algorithm
#Author: Justin Nie
#Date: 2018/1/25

from numpy import *
import operator

def classify0(input_x, dataset, labels, k):
	'''
		classify input_x into labels with training dataset
		input_x: a vector that stores the data to be classified
		dataset: a matrix that stores the data to be trained
		lablels: a vector that stores the label corresponding to dataset

	'''
	dataset_size = dataset.shape[0]				#number of samples
	diff_mat = tile(input_x, (dataset_size, 1)) - dataset
	square_diff_mat = diff_mat**2
	square_distances = square_diff_mat.sum(axis = 1)
	distances = square_distances**0.5
	#sort the distance and return the sorted indices
	sorted_indices = distances.argsort()
	
	#count the k voted labels and select the best label
	class_count = {}
	for i in range(k):
		vote_label = labels[sorted_indices[i]]
		class_count[vote_label] = class_count.get(vote_label, 0) + 1
	sorted_class_count = sorted(class_count.items(), 
		key = operator.itemgetter(1), reverse = True)
	
	return sorted_class_count[0][0]
	
def file2mat(filename):
	'''
		convert a file to a matrix
		filename: the path to the file
		return_mat: the returned data matrix
		class_label_vector: the returned label vector
	'''
	fr = open(filename)
	array_of_lines = fr.readlines()
	number_of_lines = len(array_of_lines)
	return_mat = zeros((number_of_lines, 3))
	class_label_vector = []
	index = 0
	for line in array_of_lines:
		line = line.strip()
		list_from_line = line.split('\t')
		return_mat[index, :] = list_from_line[0: 3]
		class_label_vector.append(int(list_from_line[-1]))
		index += 1

	return return_mat, class_label_vector 

def auto_norm(dataset):
	'''
		make the value normal, return normal dataset, ranges between
	    mininum values and maximum values, and minimum values
	    dataset: the data matrix to be normed
	    norm_dataset: the returned normed matrix
	    ranges: the returned vector of ranges of each feature
	    min_values: the returned vector of minimum values of each feature
	'''
	min_values = dataset.min(0)
	max_values = dataset.max(0)
	ranges = max_values - min_values

	norm_dataset = zeros(shape(dataset))
	m = dataset.shape[0]
	norm_dataset = dataset - tile(min_values, (m, 1))
	norm_dataset = norm_dataset / tile(ranges, (m, 1))

	return norm_dataset, ranges, min_values 


































