#Test for regression
#Author: Justin Nie
#Date: 2018/2/7

from numpy import *
from regression import *

dataset, labels = load_dataset('ex0.txt')
y_hat1 = standard_regression(dataset, labels)
#plot_dot(dataset, labels, y_hat1)

y_hat2 = lwlr_test(dataset, dataset, labels, 0.003)
plot_dot(dataset, labels, y_hat2)

