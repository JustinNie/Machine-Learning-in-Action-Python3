#classify 0-9 with KNN algorithm
#Author: Justin Nie
#Date: 2018/1/26

from numpy import *
from os import listdir
import knn

def image2vector(filename):
	'''convert an image to a vector'''
	vector = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		line_str = fr.readline()
		for j in range(32):
			vector[0, 32*i+j] = int(line_str[j])

	return vector

def digit_class_test():
	'''test the accuracy of the classifier'''
	digits_labels = []
	train_list = listdir('trainingDigits')
	number_samples = len(train_list)

	#get training matrix and training labels
	train_matrix = zeros((number_samples, 1024))
	for i in range(number_samples):
		filename_str = train_list[i]
		file_str = filename_str.split('.')[0]
		class_number_str = int(file_str.split('_')[0])
		digits_labels.append(class_number_str)
		train_matrix[i, :] = image2vector('trainingDigits/%s' % filename_str)

	test_list = listdir('testDigits')
	error_count = 0
	number_test = len(test_list)

	for i in range(number_test):
		filename_str = test_list[i]
		file_str = filename_str.split('.')[0]
		class_number_str = int(file_str.split('_')[0])
		vector_test = image2vector('testDigits/%s' % filename_str)

		classifier_result = knn.classify0(vector_test, train_matrix, 
			digits_labels, 3)
		if(classifier_result != class_number_str):
			error_count += 1

	error_rate = float(error_count) / float(number_test)
	print("Error rate is: " + str(error_rate))


digit_class_test()







