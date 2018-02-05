#SVM algorithm
#Author: Justin Nie
#Date: 2018/2/1

from numpy import *
from time import sleep

def load_dataset(filename):
	'''
		load the dataset and labels from filename
		dataset: the returned list of feature data
		labels: the returned list of label data
	'''
	dataset = [];	labels = []
	fr = open(filename)
	for line in fr.readlines():
		line_array = line.strip().split('\t')
		dataset.append([float(line_array[0]), float(line_array[1])])
		labels.append(float(line_array[2]))

	return dataset, labels

def select_rand(value, threshold):
	'''
		select an integer between 0 - threshold but not value 
	'''
	select_value = value
	while select_value == value:
		select_value = int(random.uniform(0, threshold))

	return select_value

def clip_alpha(alpha, high, low):
	'''
		clip the alpha value between low and high
	'''
	if alpha > high:
		alpha = high
	if alpha < low:
		alpha = low

	return alpha

#version 1
def smo_simple(dataset, labels, c, tolerance, max_iteration):
	'''
		A simple version of SMO algorithm
		dataset: a list of feature data
		labels: a list of label data
		c: the threshold to cut alphas
		tolerance: tolereance to error
		max_iteration: the maximum iteration, which can be set
		b: the returned b value
		alphas: a list of alpha value
	'''
	data_matrix = mat(dataset)
	label_matrix = mat(labels).transpose()		#column vector
	m, n = shape(data_matrix)
	b = 0;	alphas = mat(zeros((m, 1)))			#column vector
	iteration = 0

	while (iteration < max_iteration):
		alpha_changed = 0
		for i in range(m):
			estimate_i = float(multiply(alphas, label_matrix).T * \
			(data_matrix * data_matrix[i, :].T)) + b
			error_i = estimate_i - float(label_matrix[i])

			if ((label_matrix[i]*error_i < -tolerance) and (alphas[i] < c)) or \
				((label_matrix[i]*error_i > tolerance) and (alphas[i] > 0)):
				j = select_rand(i, m)
				estimate_j = float(multiply(alphas, label_matrix).T * \
				(data_matrix * data_matrix[j, :].T)) + b
				error_j = estimate_j - float(label_matrix[j])
				
				#guarantee alphas stay between 0 and c
				alpha_i_old = alphas[i].copy();	alpha_j_old = alphas[j].copy()
				if (label_matrix[i] != label_matrix[j]):
					low = max(0, alphas[j] - alphas[i])
					high = min(c, c + alphas[j] - alphas[i])
				else:
					low = max(0, alphas[j] + alphas[i] - c)
					high = min(c, alphas[j] + alphas[i])
				if low == high:
					print("low == high")
					continue
				
				estimate = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - \
				data_matrix[i, :] * data_matrix[i, :].T - \
				data_matrix[j, :] * data_matrix[j, :].T
				if estimate >= 0:
					print ("estimate >= 0")
					continue
				#update alphas[i] and alphas[j]
				alphas[j] -= label_matrix[j]*(error_i-error_j) / estimate
				alphas[j] = clip_alpha(alphas[j], high, low)
				if (abs(alphas[j] - alpha_j_old) < 0.00001):
					print("j not moving enough")
					continue
				alphas[i] += label_matrix[j] * label_matrix[i] * \
				(alpha_j_old - alphas[j])
				#select the constant value
				b1 = b - error_i - label_matrix[i]*(alphas[i]-alpha_i_old)* \
				data_matrix[i, :]*data_matrix[i, :].T - \
				label_matrix[j]*(alphas[j]-alpha_j_old)* \
				data_matrix[i, :]*data_matrix[j, :].T

				b2 = b - error_j - label_matrix[i]*(alphas[i]-alpha_i_old)* \
				data_matrix[i, :]*data_matrix[j, :].T - \
				label_matrix[j]*(alphas[j]-alpha_j_old)* \
				data_matrix[j, :]*data_matrix[j, :].T

				if (0 < alphas[i]) and (c > alphas[i]):
					b = b1
				elif (0 < alphas[j]) and (c > alphas[j]):
					b = b2
				else:
					b = (b1 + b2) / 2.0
				alpha_changed += 1
				print("Iteration: %d i: %d, pairs changed %d" %(iteration, i, 
					alpha_changed))
		if (alpha_changed == 0):
			iteration += 1
		else:
			iteration = 0
		print("Iteration number: %d" % iteration)

	ws = zeros((n, 1))
	for i in range(m):
		ws += multiply(alphas[i] * label_matrix[i], data_matrix[i, :].T)

	return b, ws, alphas

#===============================================================================
#version2
class OptStruct():
	"""docstring for OptStruct"""
	def __init__(self, data_matrix, labels, c, tolerance):
		self.x = data_matrix
		self.label_matrix = labels
		self.c = c
		self.tolerance = tolerance
		self.m = shape(data_matrix)[0]
		self.alphas = mat(zeros((self.m, 1)))
		self.b = 0
		self.error_cache = mat(zeros((self.m, 2)))

def cal_error(os, k):
	'''
		calculate error of k
	'''
	estimate_k = float(multiply(os.alphas, os.label_matrix).T * 
		(os.x * os.x[k, :].T)) + os.b
	error_k = estimate_k - float(os.label_matrix[k])

	return error_k

def select_j(i, os, error_i):
	'''
		this is the second choice -heurstic, and calcs error of j
	'''
	max_k = -1;	max_delta_error = 0;	error_j = 0
	os.error_cache[i] = [1, error_i]
	valid_error_cache = nonzero(os.error_cache[:, 0].A)[0]

	if (len(valid_error_cache) > 1):
		for k in valid_error_cache:
			if k == i:
				continue
			error_k = cal_error(os, k)
			delta_error = abs(error_i - error_k)
			if (delta_error > max_delta_error):
				max_k = k;	max_delta_error = delta_error;	error_j = error_k
		return max_k, error_j
	else:
		j = select_rand(i, os.m)
		error_j = cal_error(os, j)
	return j, error_j

def update_error(os, k):
	'''
		after any alpha has changed update the new value in the cache
	'''
	error_k = cal_error(os, k)
	os.error_cache[k] = [1, error_k]

def inner_loop(i, os):
	'''
		the inner loop of the second version of smo algorithm
	'''
	error_i = cal_error_k(os, i)
	if ((os.label_matrix[i]*error_i < -os.tolerance) and (os.alphas[i] < os.c)) or\
		((os.label_matrix[i]*error_i > os.tolerance) and (os.alphas[i] > 0)):
		j, error_j = select_j(i, os, error_i)
		alpha_i_old = os.alphas[i].copy();	alpha_j_old = os.alphas[j].copy()
		
		if (os.label_matrix[i] != os.label_matrix[j]):
			low = max(0, os.alphas[j] - os.alphas[i])
			high = min(os.c, os.c + os.alphas[j] - os.alphas[i])
		else:
			low = max(0, os.alphas[j] + os.alphas[i] - os.c)
			high = min(os.c, os.alphas[j] + os.alphas[i])
		if low == high:
			print("low = high");	return 0
		
		estimate = 2.0 * os.x[i, :]*os.x[j, :].T - os.x[i, :]*os.x[i, :].T - \
		os.x[j, :]*os.x[j, :].T
		if estimate >= 0:
			print("estimate >= 0");	return 0

		os.alphas[j] -= os.label_matrix[j] * (error_i-error_j) / estimate
		os.alphas[j] = clip_alpha(os.alphas[j], high, low)
		update_error(os, j)
		if (abs(os.alphas[j] - alpha_j_old) < 0.00001):
			print("j not moving enough");	return 0

		os.alphas[i] += os.label_matrix[j]*os.label_matrix[i] * \
		(alpha_j_old - os.alphas[j])
		update_error(os, i)

		b1 = os.b - error_i - os.label_matrix[i] * (os.alphas[i]-alpha_i_old) * \
		os.x[i, :]*os.x[i, :].T - os.label_matrix[j] * (os.alphas[j]-alpha_j_old)*\
		os.x[i, :] * os.x[j, :].T
		b2 = os.b - error_j - os.label_matrix[i] * (os.alphas[i]-alpha_i_old) * \
		os.x[i, :]*os.x[j, :].T - os.label_matrix[j] * (os.alphas[j]-alpha_j_old)*\
		os.x[j, :] * os.x[j, :].T
		if (os.alphas[i] > 0) and (os.alphas[i] < os.c):
			os.b = b1
		elif (os.alphas[j] > 0) and (os.alphas[j] < os.c):
			os.b = b2
		else:
			os.b = (b1+b2) / 2.0
		return 1
	else:
		return 0

def smo_simple2(dataset, labels, c, tolerance, max_iteration):
	'''
		full platt smo version but without kernels
	'''
	os = OptStruct(mat(dataset), mat(labels).transpose(), c, tolerance)
	iteration = 0;	alpha_changed = 0
	entire_set = True

	while (iteration < max_iteration) and ((alpha_changed > 0) or (entire_set)):
		alpha_changed = 0
		if entire_set:			#go over all
			for i in range(os.m):
				alpha_changed += inner_loop(i, os)
				print("Full set, iteration: %d i: %d, pairs changed %d" 
					% (iteration, i, alpha_changed))
			iteration += 1
		else:	#go over non-bound (railed) alphas
			non_bounds = nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
			for i in non_bounds:
				alpha_changed += inner_loop(i, os)
				print("Non nound, iteration: %d i: %d, pairs changed %d" 
					% (iteration, i, alpha_changed))
			iteration += 1
		
		if entire_set:
			entire_set = False
		elif alpha_changed == 0:
			entire_set = True
		print("Iteration number: %d" %iteration)

	m, n = shape(dataset)
	ws = zeros((n, 1))
	for i in range(m):
		ws += multiply(os.alphas[i] * os.label_matrix[i], os.x[i, :].T)
	os.ws = ws

	return os.b, os.ws, os.alphas

#===============================================================================
#version 3 with kernels
def kernel_trans(data_matrix, data_matrix_i, k_tuple):
	'''
		transpose to the kernel
	'''
	m, n = shape(data_matrix)
	kernel = mat(zeros((m, 1)))
	if k_tuple[0] == 'lin':
		kernel = data_matrix * data_matrix_i.T
	elif k_tuple[0] == 'rbf':
		for j in range(m):
			delta_row = data_matrix[j, :] - data_matrix_i
			kernel[j] = delta_row * delta_row.T
		kernel = exp(kernel / (-1 * k_tuple[1]**2))
	else:
		raise NameError("wrong kernel")
	return kernel

class OptStructk():
	"""docstring for OptStruct"""
	def __init__(self, data_matrix, labels, c, tolerance, k_tuple):
		self.x = data_matrix
		self.label_matrix = labels
		self.c = c
		self.tolerance = tolerance
		self.m = shape(data_matrix)[0]
		self.alphas = mat(zeros((self.m, 1)))
		self.b = 0
		self.error_cache = mat(zeros((self.m, 2)))
		self.kernels = mat(zeros((self.m, self.m)))
		for i in range(self.m):
			self.kernels[:, i] = kernel_trans(self.x, self.x[i, :], k_tuple)

def cal_error_k(os, k):
	'''
		calculate error of k
	'''
	estimate_k = float(multiply(os.alphas, os.label_matrix).T * 
		os.kernels[:, k] + os.b)
	error_k = estimate_k - float(os.label_matrix[k])

	return error_k

def update_error_k(os, k):
	'''
		after any alpha has changed update the new value in the cache
	'''
	error_k = cal_error_k(os, k)
	os.error_cache[k] = [1, error_k]

def select_jk(i, os, error_i):
	'''
		this is the second choice -heurstic, and calcs error of j
	'''
	max_k = -1;	max_delta_error = 0;	error_j = 0
	os.error_cache[i] = [1, error_i]
	valid_error_cache = nonzero(os.error_cache[:, 0].A)[0]

	if (len(valid_error_cache) > 1):
		for k in valid_error_cache:
			if k == i:
				continue
			error_k = cal_error_k(os, k)
			delta_error = abs(error_i - error_k)
			if (delta_error > max_delta_error):
				max_k = k;	max_delta_error = delta_error;	error_j = error_k
		return max_k, error_j
	else:
		j = select_rand(i, os.m)
		error_j = cal_error_k(os, j)
	return j, error_j


def inner_loop_k(i, os):
	'''
		the inner loop of the second version of smo algorithm
	'''
	error_i = cal_error_k(os, i)
	if ((os.label_matrix[i]*error_i < -os.tolerance) and (os.alphas[i] < os.c)) or\
		((os.label_matrix[i]*error_i > os.tolerance) and (os.alphas[i] > 0)):
		j, error_j = select_jk(i, os, error_i)
		alpha_i_old = os.alphas[i].copy();	alpha_j_old = os.alphas[j].copy()
		
		if (os.label_matrix[i] != os.label_matrix[j]):
			low = max(0, os.alphas[j] - os.alphas[i])
			high = min(os.c, os.c + os.alphas[j] - os.alphas[i])
		else:
			low = max(0, os.alphas[j] + os.alphas[i] - os.c)
			high = min(os.c, os.alphas[j] + os.alphas[i])
		if low == high:
			print("low = high");	return 0
		
		estimate = 2.0 * os.kernels[i, j] - os.kernels[i, i] - os.kernels[j, j]
		if estimate >= 0:
			print("estimate >= 0");	return 0

		os.alphas[j] -= os.label_matrix[j] * (error_i-error_j) / estimate
		os.alphas[j] = clip_alpha(os.alphas[j], high, low)
		update_error_k(os, j)
		if (abs(os.alphas[j] - alpha_j_old) < 0.00001):
			print("j not moving enough");	return 0

		os.alphas[i] += os.label_matrix[j]*os.label_matrix[i] * \
		(alpha_j_old - os.alphas[j])
		update_error_k(os, i)

		b1 = os.b - error_i - os.label_matrix[i] * (os.alphas[i]-alpha_i_old) * \
		os.kernels[i, i] - os.label_matrix[j] * (os.alphas[j]-alpha_j_old) * \
		os.kernels[i, j]
		b2 = os.b - error_j - os.label_matrix[i] * (os.alphas[i]-alpha_i_old) * \
		os.kernels[i, j] - os.label_matrix[j] * (os.alphas[j]-alpha_j_old) * \
		os.kernels[j, j]
		
		if (os.alphas[i] > 0) and (os.alphas[i] < os.c):
			os.b = b1
		elif (os.alphas[j] > 0) and (os.alphas[j] < os.c):
			os.b = b2
		else:
			os.b = (b1+b2) / 2.0
		return 1
	
	else:
		return 0


def smo_simple_k(dataset, labels, c, tolerance, max_iteration, 
	k_tuple = ('lin', 0)):
	'''
		full platt smo version but without kernels
	'''
	os = OptStructk(mat(dataset), mat(labels).transpose(), c, tolerance, k_tuple)
	iteration = 0;	alpha_changed = 0
	entire_set = True

	while (iteration < max_iteration) and ((alpha_changed > 0) or (entire_set)):
		alpha_changed = 0
		if entire_set:			#go over all
			for i in range(os.m):
				alpha_changed += inner_loop_k(i, os)
				print("Full set, iteration: %d i: %d, pairs changed %d" 
					% (iteration, i, alpha_changed))
			iteration += 1
		else:	#go over non-bound (railed) alphas
			non_bounds = nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
			for i in non_bounds:
				alpha_changed += inner_loop_k(i, os)
				print("Non nound, iteration: %d i: %d, pairs changed %d" 
					% (iteration, i, alpha_changed))
			iteration += 1
		
		if (entire_set):
			entire_set = False
		elif (alpha_changed == 0):
			entire_set = True
		print("Iteration number: %d" %iteration)

	return os.b, os.alphas


def classify_smo(input_x, ws, b):
	'''
		classify a vector using parameters ws and b
	'''
	label_matrix = input_x * mat(ws) + b
	label_x = label_matrix[0][0]
	return label_x















































