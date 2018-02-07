#Badic functions about regression
#Author: Justin Nie
#Date: 2018/2/7

from numpy import *

def load_dataset(filename):
	'''
		load a dataset
	'''
	dataset = [];	labels = []
	fr = open(filename)
	for line in fr.readlines():
		line = line.strip().split('\t')
		line_array = []
		for item in line:
			line_array.append(float(item))
		
		dataset.append(line_array[0: -1])
		labels.append(line_array[-1])
	return dataset, labels

def standard_regression(dataset, labels):
	'''
		standard regression, return parameters ws
	'''
	data_matrix = mat(dataset);	label_matrix = mat(labels).T

	xt_x = data_matrix.T * data_matrix
	if linalg.det(xt_x) == 0.0:
		print("This matrix is singular, cannot do inverse.")
		return
	ws = xt_x.I * (data_matrix.T * label_matrix)
	y_hat = data_matrix * ws

	return ws, y_hat

def plot_dot(dataset, labels, y_hat):
	import matplotlib.pyplot as plt
	data_matrix = mat(dataset);	label_matrix = mat(labels).T
	m, n = shape(data_matrix)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(data_matrix[:, 1].flatten().A[0], label_matrix[:, 0].flatten().A[0], 
		c='red')

	x_copy = data_matrix.copy()
	sort_index = x_copy[:, 1].argsort(0)
	x_sort = x_copy[sort_index][:, 0, :]
	ax.plot(x_sort[:, 1], y_hat[sort_index, 0])

	plt.show()

def local_regression(test_point, x_array, y_array, k=1.0):
	'''
		local weighted linear regression
	'''
	x_mat = mat(x_array);	y_mat = mat(y_array).T
	m, n = shape(x_array)
	weights = mat(eye((m)))

	for j in range(m):
		diff_mat = test_point - x_mat[j, :]
		weights[j, j] = exp(diff_mat*diff_mat.T / (-2.0 * k**2))
	xt_x = x_mat.T * (weights * x_mat)

	if linalg.det(xt_x) == 0:
		print("This matrix is not singular, cannot do inverse.")
		return
	ws = xt_x.I * (x_mat.T * (weights * y_mat))

	return test_point*ws

def lwlr_test(test_array, x_array, y_array, k=1.0):
	'''

	'''
	m, n = shape(x_array)
	y_hat = mat(zeros((m, 1)))
	for i in range(m):
		y_hat[i, 0] = local_regression(test_array[i], x_array, y_array, k)

	return y_hat

def rss_error(y_array, y_hat):
	'''
		describe the error of our estimate
	'''
	y_hat_array = array(y_hat[0])
	return ((y_array - y_hat_array)**2).sum()

def ridge_regression(x_array, y_array, lam=0.2):
	'''
		regression with ridge
	'''
	x_mat = mat(x_array);	y_mat = mat(y_array).T
	xt_x = x_mat.T * x_mat
	denom = xt_x + eye(shape(x_mat)[1]) * lam

	if linalg.det(xt_x) == 0.0:
		print("This matrix is not singular, cannot do inverse.")
		return
	ws = denom.I * (x_mat.T * y_mat)
	y_hat = x_mat * ws

	return ws, y_hat

def ridge_test(x_array, y_array):
	'''
		test based on regression with ridge
	'''
	x_mat = mat(x_array);	y_mat = mat(y_array).T
	y_mean = mean(y_mat, 0)
	y_mat = y_mat - y_mean
	x_means = mean(x_mat, 0)
	x_var = var(x_mat, 0)
	x_mat = (x_mat - x_means) / x_var

	number = 30
	w_mat = zeros((number, shape(x_mat)[1]))
	for i in range(number):
		ws, y_hat = ridge_regression(x_array, y_array, exp(i-10))
		w_mat[i, :] = ws.T

	return w_mat

def regularize(x_mat):
	'''
		regularize the matrix x
	'''
	x_mat_copy = x_mat.copy()
	x_means = mean(x_mat_copy, 0)
	x_var = var(x_mat, 0)
	x_mat_copy = (x_mat_copy - x_means) / x_var

	return x_mat_copy

def stage_wise(x_array, y_array, eps=0.01, iteration=100):
	'''
		Forward stagewise linear regression
	'''
	x_mat = mat(x_array);	y_mat = mat(y_array).T
	y_mean = mean(y_mat, 0)
	y_mat = y_mat - y_mean
	x_mat = regularize(x_mat)
	m, n = shape(x_mat)
	return_mat = zeros((iteration, n))

	ws = zeros((n, 1));	ws_test = ws.copy();	ws_max = ws.copy()
	for i in range(iteration):
		lowest_error = inf
		for j in range(n):
			for sign in [-1, 1]:
				ws_test = ws.copy()
				ws_test[j] += eps*sign
				y_test = x_mat * ws_test
				rss_errors = rss_error(y_array, y_test)
				if rss_errors < lowest_error:
					lowest_error = rss_errors
					ws_max = ws_test
		ws = ws_max.copy()
		return_mat[i, :] = ws.T

	return return_mat





































