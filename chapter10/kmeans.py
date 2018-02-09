#Basic functions about k-means cluster algorithm.
#Author: Justin Nie
#Date: 2018/2/9

from numpy import *


def load_dataset(filename):
	'''
		load dataset from filename, return a list of dataset
	'''
	fr = open(filename);	dataset = []
	for line in fr.readlines():
		line = line.strip().split('\t')
		line_list = []
		for item in line:
			line_list.append(float(item))
		dataset.append(line_list)

	return dataset

def plot2dim(dataset, centroids, cluster):
	'''
		plot points of (x, y) with cluster and centroids
	'''
	import matplotlib.pyplot as plt
	data_matrix = mat(dataset)
	number_centroid = shape(centroids)[0]
	list_color = ['b', 'r', 'c', 'g', 'k', 'm', 'w', 'y']
	if number_centroid > len(list_color):
		print("Too many centroids.")
		return None

	fig = plt.figure()
	for i in range(number_centroid):
		pos_index = nonzero(cluster[:, 0].A == i)[0]
		x = data_matrix[pos_index, 0].flatten().A[0]
		y = data_matrix[pos_index, 1].flatten().A[0]

		ax = fig.add_subplot(111)
		ax.scatter(x, y, c=list_color[i], marker='x')
		ax.scatter(centroids[i, 0], centroids[i, 1], c=list_color[i], s=40)

	plt.show()

def euclidean_distance(vector_a, vector_b):
	'''
		calculate the euclidean distance of vector a and vector b
	'''
	return sqrt(sum(power(vector_a - vector_b, 2)))

def rand_centroids(dataset, k):
	'''
		creates a set of k random centroids for a given dataset
	'''
	m, n = shape(dataset)
	data_matrix = mat(dataset)
	centroids = mat(zeros((k, n)))

	for j in range(n):
		min_value = min(data_matrix[:, j])[0, 0]
		max_value = max(data_matrix[:, j])[0, 0]
		range_value = max_value - min_value
		centroids[:, j] = min_value + range_value * random.rand(k, 1)

	return centroids

def kmeans(dataset, k, dist_type=euclidean_distance, create_cent = rand_centroids):
	'''
		k-means cluster algorithm
	'''
	data_matrix = mat(dataset)
	m, n = shape(data_matrix)
	centroids = create_cent(dataset, k)
	cluster = mat(zeros((m, 2)))
	cluster -= 1
	cluster_changed = True

	while cluster_changed:
		cluster_changed = False
		for i in range(m):		#for each example
			min_dist = inf;	min_index = -1
			for j in range(k):		#for each centroid
				dist = dist_type(centroids[j], data_matrix[i])
				if dist < min_dist:
					min_dist = dist
					min_index = j
			
			if cluster[i, 0] != min_index:
				cluster_changed = True
			cluster[i, :] = min_index, min_dist**2

		for cent in range(k):
			pos_index = nonzero(cluster[:, 0].A == cent)[0]
			pos_data = data_matrix[pos_index]
			centroids[cent, :] = mean(pos_data, axis=0)

	return centroids, cluster

def bisect_kmeans(dataset, k, dist_type=euclidean_distance):
	'''
		bisect k-means algorithm
	'''
	#initialize one cluster
	m, n = shape(dataset)
	data_matrix = mat(dataset)
	cluster = mat(zeros((m, 2)))
	centroid = mean(data_matrix, axis=0).tolist()[0]
	centroids = [centroid]

	for j in range(m):
		cluster[j, 1] = dist_type(data_matrix[j], mat(centroid)) **2

	#split the cluster until to k clusters
	while shape(centroids)[0] < k:
		lowest_sse = inf
		#split each cluster
		for i in range(shape(centroids)[0]):
			pos_index = nonzero(cluster[:, 0].A == i)[0]
			pos_data = data_matrix[pos_index]
			split_cent, split_cluster = kmeans(pos_data, 2, dist_type)
			split_sse = sum(split_cluster[:, 1])
			old_sse = sum(cluster[nonzero(cluster[:, 0].A != i)[0], 1])

			if (old_sse + split_sse) < lowest_sse:
				best_split_index = i
				best_centroids = split_cent.A
				best_cluster = split_cluster.copy()
				lowest_sse = old_sse + split_sse

		#add the new cluster to the result
		best_cluster[nonzero(best_cluster[:, 0].A == 1)[0], 0] = shape(centroids)[0]
		best_cluster[nonzero(best_cluster[:, 0].A == 0)[0], 0] = best_split_index
		cluster[nonzero(cluster[:, 0].A == best_split_index)[0], :] = best_cluster

		centroids[best_split_index] = best_centroids[0]
		centroids.append(best_centroids[1])

	return mat(centroids), cluster












































