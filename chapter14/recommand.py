#A recommandation system for restaurant
#Author: Justin Nie
#Date: 2018/2/16

from numpy import *
import time
from svd import *

def standrand_estimate(data_mat, user_index, similarity_measure, item):
	'''
		Standrand estimate measure
	'''
	number_feature = shape(data_mat)[1]
	total_similarity = 0.0;		total_rate = 0.0
	for feature in range(number_feature):
		user_rating = data_mat[user_index, feature]
		if user_rating == 0:
			continue
		overlap_index = nonzero(logical_and(data_mat[:, feature].A > 0, \
			data_mat[:, item].A > 0))[0]
		if len(overlap_index) == 0:
			similarity = 0.0
		else:
			similarity = similarity_measure(data_mat[overlap_index, item], \
				data_mat[overlap_index, feature])

		total_similarity += similarity
		total_rate += similarity * user_rating

	if total_similarity == 0:
		return 0
	else:
		return total_rate / total_similarity

def svd_estimate(data_mat, user_index, similarity_measure, item):
	'''
		SVD estimate measure
	'''
	number_feature = shape(data_mat)[1]
	total_similarity = 0.0;		total_rate = 0.0
	u, sigma, vt, new_data_mat = svd(data_mat)
	formed_items = data_mat.T * u[:, :len(sigma)] * sigma.I
	for feature in range(number_feature):
		user_rating = data_mat[user_index, feature]
		if user_rating == 0 or feature == item:
			continue
		similarity = similarity_measure(formed_items[feature, :].T, \
			formed_items[item, :].T)
		total_similarity += similarity
		total_rate += similarity * user_rating
	if total_similarity == 0:
		return 0
	else:
		return total_rate / total_similarity

def recommand(data_mat, user_index, recommand_number=3, \
	similarity_measure=cosine_similarity, estimate_measure=standrand_estimate):
	'''
		recommand some food for user
	'''
	unrated_items = nonzero(data_mat[user_index, :].A == 0)[1]
	if len(unrated_items) == 0:
		print("you rated all items.")
		return 0
	
	items_score = []
	for item in unrated_items:
		estimate_score = estimate_measure(data_mat, user_index, \
			similarity_measure, item)
		items_score.append((item, estimate_score))
	recommand_items = sorted(items_score, \
		key=lambda index: index[1], reverse=True)[:recommand_number]

	return recommand_items


dataset = load_simdata2()
data_mat = mat(dataset)

print(data_mat)

user_index = 2

start1 = time.clock()
print("Recommand for user %d" % user_index)
print(recommand(data_mat, user_index))
print(recommand(data_mat, user_index, similarity_measure = pearson_similarity))
print(recommand(data_mat, user_index, similarity_measure = euclidian_similarity))
end1 = time.clock()
print("Time: %f s" % (end1 - start1))

start2 = time.clock()
print("Recommand for user %d" % user_index)
print(recommand(data_mat, user_index, estimate_measure = svd_estimate))
print(recommand(data_mat, user_index, estimate_measure = svd_estimate, \
	similarity_measure = pearson_similarity))
print(recommand(data_mat, user_index, estimate_measure = svd_estimate, \
	similarity_measure = euclidian_similarity))
end2 = time.clock()
print("Time: %f s" % (end2 - start2))











































