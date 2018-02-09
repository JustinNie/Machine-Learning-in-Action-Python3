#Test for k-means cluster algorithm.
#Author: Justin Nie
#Date: 2018/2/9

from numpy import *
from kmeans import *

'''
#k-means
dataset = load_dataset('testSet.txt')
data_matrix = mat(dataset)
centroids, cluster = kmeans(dataset, 4)
plot2dim(dataset, centroids, cluster)
print(centroids)
print(cluster)
'''

#bisect k-means
dataset = load_dataset('testSet2.txt')
data_matrix = mat(dataset)
centroids, cluster =  bisect_kmeans(dataset, 3)
plot2dim(dataset, centroids, cluster)
print(centroids)
print(cluster)


