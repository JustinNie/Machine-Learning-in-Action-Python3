#Test for PCA
#Author: Justin Nie
#Date: 2018/2/15

from numpy import *
from pca import *

dataset = load_dataset('testSet.txt')
data_mat = mat(dataset)
low_data_mat, new_data_mat = pca(data_mat, 1)
plot_points(data_mat, new_data_mat)
