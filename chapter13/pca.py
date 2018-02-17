#PCA
#Author: Justin Nie
#Date: 2018/2/15

from numpy import *

def load_dataset(filename, delim = '\t'):
	'''
		return a list of two dimension
	'''
	fr = open(filename)
	dataset = []
	for line in fr.readlines():
		line = line.strip().split(delim)
		line_array = []
		for item in line:
			line_array.append(float(item))
		dataset.append(line_array)

	return dataset

def replace_nan(data_mat):
	'''
		replace Nan with mean value
	'''
	number_feature = shape(data_mat)[1]
	for feature in range(number_feature):
		mean_value = mean(data_mat[nonzero(~isnan(
			data_mat[:, feature].A))[0], feature])
		data_mat[nonzero(isnan(data_mat[:, feature].A))[0], feature] = mean_value

	return data_mat

def check_eigen(data_mat, number_feature=20):
	'''
		check eigen values and vectors
	'''

	mean_values = mean(data_mat, axis=0)
	mean_removed = data_mat - mean_values
	cov_mat = cov(mean_removed, rowvar=0)

	eigen_values, eigen_vectors = linalg.eig(mat(cov_mat))
	eigen_values_index = argsort(eigen_values)
	eigen_values_index = eigen_values_index[::-1]
	sorted_values = eigen_values[eigen_values_index]
	total = sum(sorted_values)
	
	add_values = []
	add_values.append(sorted_values[0])
	for i in range(1, len(sorted_values)):
		add_values.append(add_values[i-1] + sorted_values[i])

	var_percentage = add_values/total * 100

	import matplotlib
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(range(1, number_feature+1), 
		var_percentage[:number_feature], marker='^')
	plt.xlabel("Principal Component Number")
	plt.ylabel("Percentage of Variance")
	plt.show()

	return eigen_values, eigen_vectors 

def pca(data_mat, number_feature=9999999):
	'''
		PCA algorithm
	'''
	mean_values = mean(data_mat, axis=0)
	mean_removed = data_mat - mean_values
	cov_mat = cov(mean_removed, rowvar=0)

	eigen_values, eigen_vectors = linalg.eig(mat(cov_mat))
	eigen_values_index = argsort(eigen_values)
	eigen_values_index = eigen_values_index[: -(number_feature+1): -1]
	eigen_vectors = eigen_vectors[:, eigen_values_index]

	low_data_mat = mean_removed * eigen_vectors
	new_data_mat = (low_data_mat * eigen_vectors.T) + mean_values

	return low_data_mat, new_data_mat


def plot_points(data_mat, new_data_mat):
	'''
		plot the low dimension data and the reconstructed data
	'''
	import matplotlib
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(data_mat[:, 0].flatten().A[0], 
		data_mat[:, 1].flatten().A[0], marker='^', s=10)
	ax.scatter(new_data_mat[:, 0].flatten().A[0], 
		new_data_mat[:, 1].flatten().A[0], marker='o', s=5, c='red')
	plt.show()








































