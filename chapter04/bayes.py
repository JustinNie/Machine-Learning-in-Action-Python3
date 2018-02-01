#Classify by naive bayes
#Author: Justin Nie
#Date: 2018/1/28

from numpy import *

def load_dataset():
	'''
		construst a dataset and class vector fot the classifier
		dataset: a matrix that contains m samples with some words each
		class_vector: 1 represents abusive while 0 not
	'''
	dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'], 
				]
	class_vector = [0, 1, 0, 1, 0, 1]

	return dataset, class_vector

def create_vocab_list(dataset):
	'''
	create a vocabulary list based on dataset
	'''
	vocab_list = set([])
	for document in dataset:
		vocab_list = vocab_list | set(document)

	return list(vocab_list)

def words2vector(words, vocab_list):
	'''
	convert a vector of words to a vector based on vocab_list
	'''
	vector = [0] * len(vocab_list)

	for word in words:
		if word in vocab_list:
			vector[vocab_list.index(word)] += 1

	return vector

def create_train_matrix(dataset, vocab_list):
	'''
		convert a dataset to a train matrix
	'''
	train_matrix = []
	for document in dataset:
		train_matrix.append(words2vector(document, vocab_list))

	return train_matrix



def train_nb(train_matrix, train_class):
	'''
		train a model with algorithm naive bayes
		train_matrix: the data to be trained
		train_class: the data class vector
		p_normal_vector: when the document is normal, return its vector
		p_abusive_vector: when the documnet is abusive, return its vector
		p_abusive: the abusive document ratio
	'''
	number_document = len(train_matrix)
	number_word = len(train_matrix[0])
	normal_number_vector = ones(number_word)
	abusive_number_vector = ones(number_word)
	normal_sum = 2; abusive_sum = 2
	p_abusive_number = 0

	for i in range(number_document):
		if train_class[i] == 0:
			normal_number_vector += train_matrix[i]
			normal_sum += sum(train_matrix[i])
		else:
			abusive_number_vector += train_matrix[i]
			abusive_sum += sum(train_matrix[i])

	p_normal_vector = log(normal_number_vector / float(normal_sum))
	p_abusive_vector = log(abusive_number_vector / float(abusive_sum))
	p_abusive = sum(train_class) / float(number_document)

	return p_normal_vector, p_abusive_vector, p_abusive

def classify_nb(classify_vector, p_normal_vector, p_abusive_vector, p_abusive):
	'''
		classify a vector with naive bayes with model parameters trained before
	'''
	p0 = sum(classify_vector * p_normal_vector) + log(1 - p_abusive)
	p1 = sum(classify_vector * p_abusive_vector) + log(p_abusive)
	if p0 > p1:
		return 0
	else:
		return 1


































