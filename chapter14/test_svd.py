#Test for SVD
#Author: Justin Nie
#Date: 2018/2/15

from numpy import *
from svd import *

u, sigma, vt, new_data_mat = svd([[1, 1], [7, 7]])
print("\nU: \n", u)
print("Sigma: \n", sigma)
print("VT: \n", vt)
print("New data: \n", new_data_mat)

dataset = load_simdata()
data_mat = mat(dataset)
u, sigma, vt, new_data_mat = svd(dataset)
print("\nU: \n", u)
print("Sigma: \n", sigma)
print("VT: \n", vt)
print("New data: \n", new_data_mat)

print("Euclidian similarity: ", euclidian_similarity(data_mat[:, 0], data_mat[:, 4]))
print("Euclidian similarity: ", euclidian_similarity(data_mat[:, 0], data_mat[:, 1]))
print("Pearson similarity: ", pearson_similarity(data_mat[:, 0], data_mat[:, 4]))
print("Pearson similarity: ", pearson_similarity(data_mat[:, 0], data_mat[:, 1]))
print("Cosine similarity: ", cosine_similarity(data_mat[:, 0], data_mat[:, 4]))
print("Cosine similarity: ", cosine_similarity(data_mat[:, 0], data_mat[:, 1]))








































