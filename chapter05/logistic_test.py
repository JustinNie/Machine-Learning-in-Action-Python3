#Logistic regression test
#Author: Justin Nie
#Date: 2018/1/30

from numpy import *
from logistic_regression import *

dataset, labels = load_dataset()
'''
weights = grad_ascent(dataset, labels)
plot_best_fit(weights)

weights = sto_grad_ascent0(dataset, labels)
plot_best_fit(weights)

weights = sto_grad_ascent1(dataset, labels)
plot_best_fit(weights)
'''
colic_test()
