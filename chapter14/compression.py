#Image compression with the SVD
#Author: Justin Nie
#Date: 2018/2/17

from numpy import *
from svd import *

def print_mat(data_mat, threshold=0.8):
	'''
		print the data matrix in the binary model
	'''
	for row_index in range(32):
		print('\n', end=' ')
		for column_index in range(32):
			if (float(data_mat[row_index, column_index])) > threshold:
				print(1, end=' ')
			else:
				print(0, end=' ')
	print('\n')

def image_compression(dataset, threshold):
	'''
		image compression with the SVD
	'''
	u, sigma, vt, new_data_mat = svd(dataset, threshold)

	return len(sigma), new_data_mat


dataset = load_dataset('0_5.txt')

data_mat = mat(dataset)
number, new_data_mat = image_compression(dataset, 0.6)
print("The original matrix:")
print_mat(data_mat)
print("\nThe reconstructed result with %d values:" % number)
print_mat(new_data_mat)

