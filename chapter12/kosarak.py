#An example for frequent items.
#Author: Justin Nie
#Date: 2018/2/14

from numpy import *
from fpgrowth import *
import time

import sys
sys.path.append('../Chapter11-Apriori')
import apriori as ap


parse_data = [line.split() for line in open('kosarak.dat').readlines()]
data_dict = list2dict(parse_data)

print("FP algorithm: ")
start_fp = time.clock()
tree, header_table = create_tree(data_dict, 100000)
print("Tree: ")
tree.display()

print('\nConditional Trees: ')
frequent_list = []
mine_tree(tree, header_table, 100000, set([]), frequent_list)
end_fp = time.clock()

print("\nFrequent Items: ")
for item in frequent_list:
	print(item)
print("\nFP-growth time: %f s" % (end_fp - start_fp))

#Maybe too long time
print("Apriori algorithm: ")
start_ap = time.clock()
frequent_ap, support_data = ap.apriori(parse_data, 0.0001)
end_ap = time.clock()

print("\nFrequent Items: ")
for item in frequent_list:
	print(item)

print("\nFP-growth time: %f s" % (end_fp - start_fp))
print("Apriori time: %f s" % (end_ap - start_ap))

