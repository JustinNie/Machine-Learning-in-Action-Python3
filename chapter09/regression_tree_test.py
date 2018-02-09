#Test for regression tree
#Author: Justin Nie
#Date: 2018/2/8

from numpy import *
from regression_tree import *

'''
dataset = load_dataset("ex00.txt")
data_matrix = mat(dataset)
tree = create_tree(data_matrix)
print(tree)
#plot_points(data_matrix)


dataset = load_dataset("ex0.txt")
data_matrix = mat(dataset)
tree = create_tree(data_matrix)
print(tree)
#plot_points(data_matrix)


dataset = load_dataset("ex2.txt")
data_matrix = mat(dataset)
tree = create_tree(data_matrix, ops=(0, 1))
print(tree)
#plot_points(data_matrix)

test_dataset = load_dataset("ex2test.txt")
test_matrix = mat(test_dataset)
tree = prune(tree, test_matrix)
print('\n')
print(tree)

'''

dataset = load_dataset("exp2.txt")
data_matrix = mat(dataset)
plot_points(data_matrix)
tree = create_tree(data_matrix, model_leaf, model_error, (1, 10))
print(tree)

