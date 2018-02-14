#Test for the FP growth
#Author: Justin Nie
#Date: 2018/2/11

from numpy import *
from fpgrowth import *

dataset = load_simdata()
data_dict = list2dict(dataset)

tree, header_table = create_tree(data_dict, 3.0)
print("Tree: ")
tree.display()

print('\nConditional Trees: ')
frequent_list = []
mine_tree(tree, header_table, 3, set([]), frequent_list)
print("\nFrequent Items: ")
for item in frequent_list:
	print(item)
