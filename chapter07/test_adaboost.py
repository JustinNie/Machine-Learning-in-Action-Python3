#Test for Adaboost
#Author: Justin Nie
#Date: 2018/2/5

from numpy import *
from adaboost import *

dataset, labels = load_simdata()
weak_classifiers, agg_estimate = adaboost_train(dataset, labels, 30)
print("**********************")
print(weak_classifiers)
print("**********************")
classify_result = adaboost_classify([[0, 0], [5, 5]], weak_classifiers)
print("**********************")
print(classify_result)

#A test on a difficult task
train_dataset, train_labels = load_dataset('horseColicTraining2.txt')
weak_classifiers, agg_estimate = adaboost_train(train_dataset, train_labels, 10)
print(weak_classifiers)
plot_roc(agg_estimate.T, train_labels)
test_dataset, test_labels = load_dataset('horseColicTest2.txt')
test_prediction = adaboost_classify(test_dataset, weak_classifiers)
error_array = mat(ones((len(test_prediction), 1)))
error_count = error_array[test_prediction != mat(test_labels).transpose()].sum()
error_rate = error_count / float(len(test_prediction))
print("error rate: %.3f" % error_rate)


