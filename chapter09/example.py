#Example: comparing tree methods to standard regression
#Author: Justin Nie
#Date: 2018/2/8

from numpy import *
from regression_tree import *

train_data = load_dataset('bikeSpeedVsIq_train.txt')
train_matrix = mat(train_data)
test_data = load_dataset('bikeSpeedVsIq_test.txt')
test_matrix = mat(test_data)

tree = create_tree(train_matrix, ops=(1, 20))
y_hat = create_forecast(tree, test_matrix[:, 0])
print(corrcoef(y_hat, test_matrix[:, 1], rowvar=0)[0, 1])

tree1 = create_tree(train_matrix, model_leaf, model_error, (1, 20))
y_hat1 = create_forecast(tree1, test_matrix[:, 0], model_tree_eval,)
print(corrcoef(y_hat1, test_matrix[:, 1], rowvar=0)[0, 1])

y_hat2 = zeros(shape(test_matrix)[0])
ws, x, y = linear_solve(train_matrix)
for i in range(shape(test_matrix)[0]):
	y_hat2[i] = test_matrix[i, 0] * ws[1, 0] + ws[0, 0]
print(corrcoef(y_hat2, test_matrix[:, 1], rowvar=0)[0, 1])
