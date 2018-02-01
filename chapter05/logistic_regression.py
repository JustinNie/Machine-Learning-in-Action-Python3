#Logistic regression
#Author: Justin Nie
#Date: 2018/1/30

from numpy import *

def load_dataset():
	'''
		load the dataset from the data file
		dataset: the returned feature data
		labels: the returned label vector
	'''
	dataset = [];	labels = []
	fr = open('test_set.txt')
	for line in fr.readlines():
		line_vector = line.strip().split()
		dataset.append([1.0, float(line_vector[0]), float(line_vector[1])])
		labels.append(int(line_vector[2]))

	return dataset, labels

def sigmoid(x):
	'''
		the function of sigmoid
	'''
	return 1.0 / (1 + exp(-x))

def grad_ascent(dataset, labels):
	'''
		optimize the weights with gradient ascent algorithm
	'''
	data_matrix = mat(dataset)
	label_matrix = mat(labels).transpose()
	m, n = shape(data_matrix)
	alpha = 0.001
	max_cycle = 500
	weights = ones((n, 1))

	for k in range(max_cycle):
		h = sigmoid(data_matrix * weights)
		error = label_matrix - h
		weights += alpha * data_matrix.transpose() * error

	return weights

def sto_grad_ascent0(dataset, labels):
	'''
		stochastic gradient ascent algorithm 0
	'''
	data_matrix = mat(dataset)
	m, n = shape(dataset)
	alpha = 0.01
	weights = ones((n, 1))
	for i in range(m):
		h = sigmoid(sum(data_matrix[i] * weights))
		error = labels[i] - h
		weights += alpha * error * data_matrix[i].transpose()

	return weights

def sto_grad_ascent1(dataset, labels, iteration = 150):
	'''
		stochastic gradient ascent algorithm 1
	'''
	data_matrix = mat(dataset)
	m, n = shape(dataset)
	weights = ones((n, 1))

	for j in range(iteration):
		data_index = list(range(m))
		for i in range(m):
			alpha = 4 / (1.0 + j + i) + 0.01
			rand = int(random.uniform(0, len(data_index)))
			rand_index = data_index[rand]
			h = sigmoid(sum(data_matrix[rand_index] * weights))
			error = labels[rand_index] - h
			weights += alpha * error * data_matrix[rand_index].transpose()
			del(data_index[rand])

	return weights



def plot_best_fit(weights):
	'''
		plot the best fit in the figure
	'''
	import matplotlib.pyplot as plt

	dataset, labels = load_dataset()
	data_array = array(dataset)
	n = shape(data_array)[0]
	x_cord1 = [];	y_cord1 = []
	x_cord2 = [];	y_cord2 = []

	for i in range(n):
		if int(labels[i]) == 1:
			x_cord1.append(data_array[i, 1])
			y_cord1.append(data_array[i, 2])
		else:
			x_cord2.append(data_array[i, 1])
			y_cord2.append(data_array[i, 2])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
	ax.scatter(x_cord2, y_cord2, s=30, c='green')

	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1]*x) / weights[2]
	ax.plot(x, y)
	plt.xlabel('X1');	plt.ylabel('X2')
	plt.show()

def classify_log(input_x, weights):
	'''
		classify an input
	'''
	prob = sigmoid(sum(input_x * weights))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0

def colic_test():
	'''
		test colic and count its accuracy
		error_rate: the returned test error rate
	'''
	fr_train = open('horseColicTraining.txt')
	fr_test = open('horseColicTest.txt')
	train_set = [];	train_labels = []

	for line in fr_train.readlines():
		current_line = line.strip().split('\t')
		line_feature = []
		for i in range(21):
			line_feature.append(float(current_line[i]))
		train_set.append(line_feature)
		train_labels.append(float(current_line[21]))
	train_weights = sto_grad_ascent1(array(train_set), train_labels)

	error_count = 0;	test_number = 0
	test_set = [];		test_labels = []

	for line in fr_test.readlines():
		current_line = line.strip().split('\t')
		line_feature = []
		for i in range(21):
			line_feature.append(float(current_line[i]))
		test_set.append(line_feature)
		train_labels.append(float(current_line[21]))

		if int(classify_log(line_feature, train_weights)) != int(
			current_line[21]):
			error_count += 1
		test_number += 1

	error_rate = error_count / float(test_number)
	print("error rate: %f" %error_rate)


































