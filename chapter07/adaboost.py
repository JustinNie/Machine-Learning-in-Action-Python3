#basic functions about Adaboost.
#Author: Justin Nie
#Date: 2018/2/5

from numpy import *


def load_simdata():
	'''
		get a simple dataset and class labels
	'''
	dataset = [[1.0, 2.1], 
				[2.0, 1.1], 
				[1.3, 1.0], 
				[1.0, 1.0], 
				[2.0, 1.0],
				]
	labels = [1.0, 1.0, -1.0, -1.0, 1.0]

	return dataset, labels

def stump_classify(data_matrix, i, thresh_value, thresh_seq):
	'''
		to feature index i classify the data
	'''
	return_array = ones((shape(data_matrix)[0], 1))
	if thresh_seq == 'lt':
		return_array[data_matrix[:, i] <= thresh_value] = -1.0
	else:
		return_array[data_matrix[:, i] > thresh_value] = -1.0

	return return_array

def bulid_stump(dataset, labels, d):
	'''
		a weak classifier
		d: the weight vector
		best_stump: the returned best stump information
		min_error: the returned minimum error
		best_estimate: the returned predicted values based on this weak classifier 
	'''
	data_matrix = mat(dataset);	label_matrix = mat(labels).transpose()
	m, n = shape(data_matrix);	number_step = 10.0
	best_stump = {};	best_estimate = mat(zeros((m, 1)))
	min_error = inf

	for i in range(n):
		min_value = data_matrix[:, i].min();	max_value = data_matrix[:, i].max()
		step_size = (max_value - min_value) / number_step

		for j in range(-1, int(number_step) + 1):
			for inequal in ['lt', 'gt']:
				thresh_value = min_value + step_size*float(j)
				predict_values = stump_classify(data_matrix, i, thresh_value, inequal)
				error_array = mat(ones((m, 1)))
				error_array[predict_values == label_matrix] = 0
				weighted_error = d.T * error_array

				print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted\
				 error %.3f" % (i, thresh_value, inequal, weighted_error))
				if (weighted_error < min_error):
					min_error = weighted_error
					best_estimate = predict_values.copy()
					best_stump['dim'] = i
					best_stump['thresh'] = thresh_value
					best_stump['inequal'] = inequal

	return best_stump, min_error, best_estimate

def adaboost_train(dataset, labels, iteration):
	'''

	'''
	weak_classifiers = []
	m, n = shape(dataset)
	d = mat(ones((m, 1)) / float(m))
	agg_estimate = mat(ones((m, 1)))

	for i in range(iteration):
		best_stump, error, estimate = bulid_stump(dataset, labels, d)
		print("D: ", d.T)
		alpha = float(0.5 * log((1.0 - error)/ max(error, 1e-16)))
		best_stump['alpha'] = alpha
		weak_classifiers.append(best_stump)
		print("label estimate: ", estimate.T)

		expon = multiply(-1 * alpha * mat(labels).T, estimate)
		d = multiply(d, exp(expon))
		d = d / d.sum()
		agg_estimate += alpha * estimate
		print("Aggregate estimate: ", agg_estimate)

		agg_error = multiply(sign(agg_estimate) != mat(labels).T, ones((m, 1)))
		error_rate = agg_error.sum() / m
		print("Total error: ", error_rate)
		if(error_rate == 0.0):
			break

	return weak_classifiers, agg_estimate

def adaboost_classify(input_x, weak_classifiers):
	'''
		to classify input_x with adaboost
	'''
	input_matrix = mat(input_x)
	m = shape(input_matrix)[0]
	agg_estimate = mat(zeros((m, 1)))

	for i in range(len(weak_classifiers)):
		estimate = stump_classify(input_matrix, weak_classifiers[i]['dim'], 
			weak_classifiers[i]['thresh'], weak_classifiers[i]['inequal'])
		agg_estimate += weak_classifiers[i]['alpha'] * estimate
		#print(agg_estimate)

	return sign(agg_estimate)

#******************************************
#Adaboost on a difficult dataset
def load_dataset(filename):
	'''
		load a dataset for Adaboost
	'''
	number_feature = len(open(filename).readline().split('\t'))
	dataset = [];	labels = []
	fr = open(filename)
	for line in fr.readlines():
		line_array = []
		current_line = line.strip().split('\t')
		for i in range(number_feature - 1):
			line_array.append(float(current_line[i]))
		dataset.append(line_array)
		labels.append(float(current_line[-1]))

	return dataset, labels

def plot_roc(agg_estimate, labels):
	'''
		plot the ROC curve in the training
	'''
	import matplotlib.pyplot as plt
	curve = (1.0, 1.0)
	y_sum = 0.0
	number_position = sum(array(labels) == 1.0)
	y_step = 1 / float(number_position)
	x_step = 1 / float(len(labels) - number_position)
	sorted_indicies = agg_estimate.argsort()

	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sorted_indicies.tolist()[0]:
		if labels[index] == 1.0:
			del_x = 0;	del_y = y_step
		else:
			del_x = x_step;	del_y = 0
			y_sum += curve[1]
		
		ax.plot([curve[0], curve[0] - del_x], [curve[1], curve[1] - del_y], c='b')
		curve = (curve[0] - del_x, curve[1] - del_y)
	ax.plot([0, 1], [0, 1], 'b--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.show()

	print("The area under the ROC curve is: ")
	print(y_sum*x_step)






































