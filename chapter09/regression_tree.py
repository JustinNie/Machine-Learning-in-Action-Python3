#Basic functions about regression tree
#Author: Justin Nie
#Date: 2018/2/8

from numpy import *
import matplotlib.pyplot as plt


def load_dataset(filename):
	'''
		load the dataset
	'''
	dataset = []
	fr = open(filename)
	for line in fr.readlines():
		line = line.strip().split('\t')
		line_array = []
		for item in line:
			line_array.append(float(item))

		dataset.append(line_array)

	return dataset

def binary_split(dataset, feature, value):
	'''
		split the dataset into two parts, based on feature and value
	'''
	left = dataset[nonzero(dataset[:, feature] > value)[0], :]
	right = dataset[nonzero(dataset[:, feature] <= value)[0], :]

	return left, right

def reg_leaf(dataset):
	
	return mean(dataset[:, -1])

def reg_error(dataset):
	
	return var(dataset[:, -1]) * shape(dataset)[0]

def best_split(dataset, leaf_type=reg_leaf, error_type=reg_error, ops=(1, 4)):
	'''
		return the best split feature index and value
	'''
	error_limit = ops[0];	number_limit = ops[1]
	if len(set(dataset[:, -1].T.tolist()[0])) == 1:		
		return None, leaf_type(dataset)				#exit if all values are qrual

	m, n = shape(dataset)
	error = error_type(dataset)
	best_error = inf;	best_index = 0;	best_value = 0
	
	for feature_index in range(n-1):
		for split_value in set((dataset[:, feature_index].T.A.tolist())[0]):
			left, right = binary_split(dataset, feature_index, split_value)
			if (shape(left)[0] < number_limit) or (shape(right)[0] < number_limit):
				continue	#the number of elements of the leaf nodes is too small
			
			new_error = error_type(left) + error_type(right)
			if new_error < best_error:
				best_error = new_error
				best_value = split_value
				best_index = feature_index
	
	#exit if low error reduction
	if (error - best_error) < error_limit:
		return None, leaf_type(dataset)
	left, right = binary_split(dataset, best_index, best_value)
	#exit if too small leaf nodes
	if (shape(left)[0] < number_limit) or (shape(right)[0] < number_limit):
		return None, leaf_type(dataset)

	return best_index, best_value

def create_tree(dataset, leaf_type=reg_leaf, error_type=reg_error, ops = (1, 4)):
	'''
		create the tree based on dataset
	'''
	feature, value = best_split(dataset, leaf_type, error_type, ops)
	if feature == None:
		return value				#return leaf value is nothing splited
	tree = {}
	tree['split_index'] = feature
	tree['split_value'] = value
	
	left, right = binary_split(dataset, feature, value)
	tree['right'] = create_tree(right, leaf_type, error_type, ops)
	tree['left'] = create_tree(left, leaf_type, error_type, ops)

	return tree

def plot_points(dataset):
	'''
		plot (x, y) points. The type of dataset is (m, 2) or (m, 3)
	'''
	m, n = shape(dataset)
	if n != 2 and n != 3:
		print("It's wrong type (%d %d)" % (m, n))
		return None

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataset[:, -2].flatten().A[0], dataset[:, -1].flatten().A[0], c='red')
	plt.xlabel('X')
	plt.ylabel('Y')

	plt.show()

def is_tree(object):
	
	return (type(object).__name__ == 'dict')

def get_mean(tree):
	'''
		get the mean value of a tree
	'''
	if is_tree(tree['left']):
		left_mean =  get_mean(tree['left'])
	else:
		left_mean = tree['left']

	if is_tree(tree['right']):
		right_mean = get_mean(tree['right'])
	else:
		right_mean = tree['right']

	return (left_mean + right_mean) / 2.0

def prune(tree, test_data):
	'''
		prune the tree based on test data
	'''
	if shape(test_data)[0] == 0:
		return get_mean(tree)
	test_left, test_right = binary_split(test_data, tree['split_index'], \
		tree['split_value'])
	
	if (is_tree(tree['left'])) or (is_tree(tree['right'])):
		if is_tree(tree['left']):
			tree['left'] = prune(tree['left'], test_left)
		if is_tree(tree['right']):
			tree['right'] = prune(tree['right'], test_right)
	else:
		error = sum(power(test_left[:, -1] - tree['left'], 2)) + \
				sum(power(test_right[:, -1] - tree['right'], 2))
		tree_mean = (tree['left'] + tree['right']) / 2.0
		error_merge = sum(power(test_data[:, -1] - tree_mean, 2))
		if error_merge < error:
			print("Merging...")
			return tree_mean
		else:
			return tree

	return tree

def linear_solve(dataset):
	'''
		linear regression
	'''
	m, n = shape(dataset)
	x = mat(ones((m, n)));	y = mat(ones((m, 1)))
	x[:, 1:n] = dataset[:, 0:n-1]
	y = dataset[:, -1]
	xt_x = x.T * x

	if linalg.det(xt_x) == 0:
		print("This matrix is singular, cannot do inverse.")
		return 0
	ws = xt_x.I * (x.T * y)

	return ws, x, y

def model_leaf(dataset):
	'''
		model the leaf nodes
	'''
	ws, x, y = linear_solve(dataset)
	return ws

def model_error(dataset):
	'''

	'''
	ws, x, y = linear_solve(dataset)
	y_hat = x * ws

	return sum(power(y - y_hat, 2))

def reg_tree_eval(model, input_data):
	return float(model)

def model_tree_eval(model, input_data):
	m, n = shape(input_data)
	x = mat(ones((1, n+1)))
	x[:, 1:n+1] = input_data

	return float(x*model)

def tree_forecast(tree, input_data, model_eval=reg_tree_eval):
	'''

	'''
	if not is_tree(tree):
		return model_eval(tree, input_data)
	if input_data[tree['split_index']] > tree['split_value']:
		if is_tree(tree['left']):
			return tree_forecast(tree['left'], input_data, model_eval)
		else:
			return model_eval(tree['left'], input_data)
	else:
		if is_tree(tree['right']):
			return tree_forecast(tree['right'], input_data, model_eval)
		else:
			return model_eval(tree['right'], input_data)

def create_forecast(tree, test_data, model_eval=reg_tree_eval):
	'''

	'''
	m = len(test_data)
	y_hat = mat(zeros((m, 1)))
	for i in range(m):
		y_hat[i, 0] = tree_forecast(tree, mat(test_data[i]), model_eval)

	return y_hat


















































		