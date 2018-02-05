#Test svm
#Author: Justin Nie
#Date: 2018/2/1

from numpy import *
from svm import *

filename = 'testSet.txt'
dataset, labels = load_dataset(filename)

'''
#test for SMO version1
b, ws, alphas = smo_simple(dataset, labels, 0.6, 0.001, 40)
number_vector = shape(alphas[alphas > 0])
print(number_vector)
print(b)
print(alphas[alphas > 0])
print(ws)
label_x = classify_smo(dataset[0], ws, b)
print(label_x)
print(labels[0])
print("===================================================")


#test for SMO version2 without kernels
b2, ws2, alphas2 = smo_simple2(dataset, labels, 0.6, 0.001, 40)
number_vector2 = shape(alphas2[alphas2 > 0])
print(number_vector2)
print(b2)
print(alphas2[alphas2 > 0])
print(ws2)

label_x2 = classify_smo(dataset[0], ws2, b2)
print(label_x2)
print(labels[0])
print("===================================================")
'''

#test for smo version3 with kernels

def test_rbf(k1 = 1.3):
	'''
		test for smo version3 with kernels
	'''
	dataset, labels = load_dataset('testSetRBF.txt')
	b, alphas = smo_simple_k(dataset, labels, 200, 0.0001, 10000, ('rbf', k1))
	data_matrix = mat(dataset);	label_matrix = mat(labels).transpose()
	sv_index = nonzero(alphas.A > 0)[0]
	support_vectors = data_matrix[sv_index]
	labels_sv = label_matrix[sv_index]
	print("There are %d support vectors" % shape(support_vectors)[0])

	m, n = shape(data_matrix);	error_count = 0
	for i in range(m):
		kernel_eval = kernel_trans(support_vectors, data_matrix[i, :], ('rbf', k1))
		predict = kernel_eval.T * multiply(labels_sv, alphas[sv_index]) + b
		if sign(predict) != sign(labels[i]):
			error_count += 1
	print("The training error rate is: %f" % (float(error_count) / m))

	dataset, labels = load_dataset('testSetRBF2.txt')
	data_matrix = mat(dataset);	label_matrix = mat(labels).transpose()
	m, n = shape(data_matrix);		error_count = 0
	for i in range(m):
		kernel_eval = kernel_trans(support_vectors, data_matrix[i, :], ('rbf', k1))
		predict = kernel_eval.T * multiply(labels_sv, alphas[sv_index]) + b
		if sign(predict) != sign(labels[i]):
			error_count += 1
	print("The test error rate is: %f" % (float(error_count) / m))

test_rbf()
















