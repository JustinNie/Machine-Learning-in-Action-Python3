#Example: using PCA to reduce the dimensionality of
#		  semiconductor manufacturing data
#Author: Justin Nie
#Date: 2018/2/15

from numpy import *
from pca import *

dataset = load_dataset('secom.data', ' ')
data_mat = mat(dataset)
data_mat = replace_nan(data_mat)
check_eigen(data_mat, 20)
low_data_mat, new_data_mat = pca(data_mat, 20)
print(shape(low_data_mat))
print(shape(data_mat))

