#Basic functions about Apriori algorithm.
#Author: Justin Nie
#Date: 2018/2/10

from numpy import *

def load_simdata():
	'''
		load a simple dataset, return a list.
	'''
	return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def create_class1(dataset):
	'''
		return the one item list of dataset.
	'''
	class1 = []
	for transaction in dataset:
		for item in transaction:
			if not [item] in class1:
				class1.append([item])

	class1.sort()
	frozen_set1 = []
	for item in class1:
		frozen_set1.append(frozenset(item))

	return frozen_set1

def list2set(dataset):
	'''
		convert a list of two dimension dataset to a list of set
	'''
	data_set = []
	for transaction in dataset:
		data_set.append(set(transaction))

	return data_set

def scan_dataset(data_set, classk, min_support):
	'''
		claculate the supported class k in the dataset
		this dataset is a list of set
		return the supported set and the support data
	'''
	candidates = {}
	for transaction in data_set:
		for item in classk:
			if item.issubset(transaction):
				if item not in candidates:
					candidates[item] = 1
				else:
					candidates[item] += 1

	total_number = len(data_set)
	support_items = [];		support_data = {}
	for candidate in candidates:
		support = candidates[candidate] / float(total_number)
		if support >= min_support:
			support_items.insert(0, candidate)
		support_data[candidate] = support

	return support_items, support_data

def create_classk(frequentk_1, k):
	'''
		from frequent k-1 calculate class k
	'''
	classk = []
	number = len(frequentk_1)
	for i in range(number):
		for j in range(i+1, number):
			#(i, j)s ensure scaning all the combination of two sets of frequentk_1.
			f1 = list(frequentk_1[i])[ :-1]
			f2 = list(frequentk_1[j])[ :-1]
			f1.sort();	f2.sort()

			#if all the elements except for the last one of the two sets are the same.
			#combine these two sets to a new candidates for classk.
			if f1 == f2:
				classk.append(frequentk_1[i] | frequentk_1[j])

	return classk

def apriori(dataset, min_support=0.5):
	'''
		accomplish the apriori algorithm
	'''
	class1 = create_class1(dataset)
	data_set = list2set(dataset)
	frequent1, support_data = scan_dataset(data_set, class1, min_support)
	frequent = [frequent1]

	k = 2
	while (len(frequent[k-2]) > 0):
		classk = create_classk(frequent[k-2], k)
		frequentk, classk_data = scan_dataset(data_set, classk, min_support)
		support_data.update(classk_data)
		frequent.append(frequentk)
		k += 1

	return frequent, support_data

def get_data(frequent_set, support_data):
	'''
		get the support data of the frequent set
	'''
	for (item, data) in support_data.items():
		if frequent_set == item:
			break
	return data



def calc_confidence(frequent_set, h, support_data, rules, min_confidence=0.7):
	'''
		calculating the confidence of a rule, 
		and then finding out which rules meet the minimum confidence.
	'''
	pruned_h = []
	for item in h:
		frequent_set_data = get_data(frequent_set, support_data)
		minus_data = get_data(frequent_set - item, support_data)
		confidence = frequent_set_data / minus_data
		if confidence >= min_confidence:
			print(frequent_set - item, '-->', item, 'confidence: ', confidence)
			rules.append((frequent_set - item, item, confidence))
			pruned_h.append(item)

	return pruned_h

def rules_merge(frequent_set, h, support_data, rules, min_confidence=0.7):
	'''
		generate more association rules from our initial itemset
	'''
	m = len(h[0])
	if (len(frequent_set) > (m + 1)):
		#try further merging
		hmp1 = create_classk(h, m+1)
		hmp1 = calc_confidence(frequent_set, hmp1, support_data, rules, min_confidence)
		if (len(hmp1) > 1):
			rules_merge(frequent_set, hmp1, support_data, rules, min_confidence)


def get_rules(frequent, support_data, min_confidence=0.7):
	'''
		get rules from frequent items and related support data
	'''
	rules = []
	for i in range(1, len(frequent)):
		for frequent_set in frequent[i]:
			h = [frozenset([item]) for item in frequent_set]
			if (i > 1):
				rules_merge(frequent_set, h, support_data, rules, min_confidence)
			else:
				calc_confidence(frequent_set, h, support_data, rules, min_confidence)

	return rules



































