#Naive Bayes test
#Author: Justin Nie
#Date: 2018/1/28

from numpy import *
from bayes import *

dataset, class_vector = load_dataset()
vocab_list = create_vocab_list(dataset)
words = ['problems', 'quit']
vector = words2vector(words, vocab_list)
print(vocab_list)
print(vector)

train_matrix = create_train_matrix(dataset, vocab_list)
p_normal_vector, p_abusive_vector, p_abusive = train_nb(
	train_matrix, class_vector)
print(p_normal_vector)
print(p_abusive_vector)
print(p_abusive)

result = classify_nb(vector, p_normal_vector, p_abusive_vector, p_abusive)
print(result)





















