#Decision Tree
#Author: Justin Nie
#Date: 2018/1/26

import matplotlib
from math import log

def create_dataset():
	'''create a dataset to test'''
	dataset = [[1, 1, 'yes'], 
			   [1, 1, 'yes'], 
			   [1, 0, 'no'], 
			   [0, 1, 'no'], 
			   [0, 1, 'no'], ]
	labels = ['no surfacing', 'flippers']
	return dataset, labels

def cal_shannon_entropy(dataset):
	'''calculate shannon entropy of dataset'''
	number_samples = len(dataset)
	labels_counts = {}
	for sample_vector in dataset:
		current_label = sample_vector[-1]
		if current_label not in labels_counts.keys():
			labels_counts[current_label] = 0
		labels_counts[current_label] += 1

	shannon_entropy = 0.0
	for key in labels_counts.keys():
		probability = float(labels_counts[key]) / number_samples
		shannon_entropy -= probability * log(probability, 2)

	return shannon_entropy

def split_dataset(dataset, axis, value):
	'''
		split the dataset according to the the feature
		dataset: data to be splited
		axis: the feature index
		value: the feature value
		splited_dataset: the returned splited dataset
	'''
	splited_dataset = []
	for sample_vector in dataset:
		if sample_vector[axis] == value:
			reduced_sample_vector = sample_vector[:axis]
			reduced_sample_vector.extend(sample_vector[axis+1:])
			splited_dataset.append(reduced_sample_vector)

	return splited_dataset

def best_feature_split(dataset):
	'''
		choose the best feature to split dataset
		dataset: the data matrix to be splited
		best_feature: the returned best feature index
	'''
	number_features = len(dataset[0]) - 1
	base_entropy = cal_shannon_entropy(dataset)
	best_info_gain = 0.0
	best_feature = -1

	for i in range(number_features):
		feat_list = [sample[i] for sample in dataset]
		unique_list = set(feat_list)
		new_entropy = 0.0

		for value in unique_list:
			sub_dataset = split_dataset(dataset, i, value)
			probability = len(sub_dataset) / float(len(dataset))
			new_entropy += probability * cal_shannon_entropy(sub_dataset)

		info_gain = base_entropy - new_entropy
		if(info_gain > best_info_gain):
			best_info_gain = new_entropy
			best_feature = i

	return best_feature

def majority_count(class_list):
	'''
		vote the majority of the class list
		class_list: a vector contains a feature
		sorted_class_count: the returned dictionary of the majority
			and its count
	'''
	class_count = {}
	for vote in class_list:
		if vote not in class_count.keys():
			class_count[vote] = 0
		class_count[vote] += 1
	sorted_class_count = sorted(class_count.iteritems(), 
		key = operaotr.itemgetter(1), reverse = True)
	return sorted_class_count[0][0]

def create_tree(dataset, labels):
	'''
		create a decision tree based on dataset
		tree: the returned tree represented by a dictionary
			{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
	'''
	class_list = [sample[-1] for sample in dataset]
	#when the class is the same, return the class
	if class_list.count(class_list[0]) == len(class_list):
		return class_list[0]
	#exhaust features, then return the majority class
	if len(dataset[0]) == 1:
		return majority_count(class_list)

	best_feature = best_feature_split(dataset)
	best_feature_label = labels[best_feature]
	tree = {best_feature_label: {}}
	del(labels[best_feature])

	feature_values = [sample[best_feature] for sample in dataset]
	unique_values = set(feature_values)

	for value in unique_values:
		sub_labels = labels[:]
		tree[best_feature_label][value] = create_tree(
			split_dataset(dataset, best_feature, value), sub_labels)
	return tree

def classify(input_tree, feature_labels, test_vector):
	''' '''
	first_sides = list(input_tree.keys())
	first_str = first_sides[0]
	second_dict = input_tree[first_str]
	feature_index = feature_labels.index(first_str)

	for key in second_dict.keys():
		if test_vector[feature_index] == key:
			if type(second_dict[key]).__name__ == 'dict':
				class_label = classify(second_dict[key], feature_labels, 
					test_vector)
			else:
				class_label = second_dict[key]

	return class_label

def store_tree(tree, filename):
	''''''
	import pickle
	fw = open(filename, 'wb')
	pickle.dump(tree, fw)
	fw.close()

def grab_tree(filename):
	''''''
	import pickle
	fr = open(filename, 'rb')
	return pickle.load(fr)



























