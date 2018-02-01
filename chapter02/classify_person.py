from numpy import*
import knn

def dating_class_test(dating_data, dating_labels):
	'''test the class and count the accuracy'''
	test_ratio = 0.1	#the ratio of test and the total
	norm_dating_data, ranges, min_values = knn.auto_norm(dating_data)
	dating_length = norm_dating_data.shape[0]
	number_vectors = int(dating_length*test_ratio)	#the number to be tested
	error_count = 0.0

	for i in range(number_vectors):
		classifier_result = knn.classify0(norm_dating_data[i, :], 
			norm_dating_data[number_vectors:dating_length, :], 
			dating_labels[number_vectors:dating_length], 3)
		if (classifier_result != dating_labels[i]):
			error_count += 1
	error_rate = error_count / number_vectors
	print ("Error rate is: %f" %error_rate)

def classify_person():
	'''classify a person for Helen'''
	result_list = ['not at all', 'in small doses', 'in large doses']
	percent_tats = float(input("percentage of time spent on games: "))
	miles = float(input("frequent flier miles earned per year: "))
	ice_cream = float(input("liters of ice_cream consumed per year: "))

	dating_data, dating_labels = knn.file2mat('dating_set.txt')
	norm_dating_data, ranges, min_values = knn.auto_norm(dating_data)
	input_x = [percent_tats, miles, ice_cream]
	classifier_result = knn.classify0((input_x - min_values) / ranges, 
		norm_dating_data, dating_labels, 3)
	print("You will probably like this person: ", 
		result_list[classifier_result-1])







dating_data, dating_labels = knn.file2mat('dating_set.txt')
dating_class_test(dating_data, dating_labels)

while True:
	'''test a person'''
	quit = input("Would you like to go on(yes or no): ")
	if quit == 'yes':
		break
	classify_person()















