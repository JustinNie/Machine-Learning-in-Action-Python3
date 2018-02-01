#Classify email based on naive bayes
#Author: Justin Nie
#Date: 2018/1/28
#Notice: email/spam/17.txt has some word can't be encoded.
#Notice: email/ham/6, 23.txt has some word can't be encoded.


from numpy import *
from bayes import *

def test_parse(big_string):
	'''
		parse the big string
	'''
	import re
	tokens = re.split(r'\W*', big_string)
	return [token.lower() for token in tokens if len(token) > 2]

def test_spam():
	'''

	'''
	dataset = [];	wordset = [];	class_vector = []
	for i in range(1, 26):
		document = test_parse(open('email/spam/%d.txt' %i).read())
		dataset.append(document)
		wordset.extend(document)
		class_vector.append(1)

		document = test_parse(open('email/ham/%d.txt' %i).read())
		dataset.append(document)
		wordset.extend(document)
		class_vector.append(0)

	vocab_list = create_vocab_list(dataset)
	train_index_set = list(range(50))
	test_index_set = []
	for i in range(10):
		rand_index = int(random.uniform(0, len(train_index_set)))
		test_index_set.append(train_index_set[rand_index])
		del(train_index_set[rand_index])

	train_matrix = [];	train_classes = []
	for i in train_index_set:
		train_matrix.append(words2vector(dataset[i], vocab_list))
		train_classes.append(class_vector[i])

	error_count = 0
	p_ham_vector, p_spam_vector, p_spam = train_nb(train_matrix, train_classes)
	for i in test_index_set:
		word_vector = words2vector(dataset[i], vocab_list)
		if class_vector[i] != classify_nb(word_vector, 
			p_ham_vector, p_spam_vector, p_spam):
			error_count += 1
			print("error email: number %d" %i)
			print(dataset[i])

	error_rate = float(error_count) / 10
	print("error rate: %f" %error_rate)


#main program
test_spam()


























