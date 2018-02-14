#Basic functions about the FP growth
#Author: Justin Nie
#Date: 2018/2/11
#Notice: Because of the type 'dictionary' is different in python3, 
#		 so the result is a little different.

from numpy import *

class TreeNode():
	"""docstring for TreeNode"""
	def __init__(self, name, count, parent_node):
		self.name = name
		self.count = count
		self.parent_node = parent_node

		self.node_link = None
		self.children = {}

	def inc(self, count_inc):
		self.count += count_inc

	def display(self, index = 1):
		'''
			display the tree
		'''
		print('  '*index, self.name, ' ', self.count)
		for child in self.children.values():
			child.display(index+1)
		

def load_simdata():
	'''
		return a simple 2-dim list of data
	'''
	simple_data = [['r', 'z', 'h', 'j', 'p'], 
				   ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
				   ['z'],
				   ['r', 'x', 'n', 'o', 's'],
				   ['y', 'r', 'x', 'z', 'q', 't', 'p'],
				   ['y', 'z', 'x', 'e', 'q', 's', 't', 'm'],
				  ]
	return simple_data

def list2dict(dataset):
	'''
		convert a list of list to a dictionary
	'''
	data_dict = {}
	for trans in dataset:
		data_dict[frozenset(trans)] = 1

	return data_dict

def update_header(test_node, target_node):
	'''
		update the header table
	'''
	while(test_node.node_link != None):
		test_node = test_node.node_link
	test_node.node_link = target_node

def update_tree(ordered_items, tree, header_table, count):
	'''
		update the tree according to the ordered items
	'''
	if ordered_items[0] in tree.children:
		tree.children[ordered_items[0]].inc(count)
	else:
		tree.children[ordered_items[0]] = TreeNode(ordered_items[0], count, tree)
		if header_table[ordered_items[0]][1] == None:
			header_table[ordered_items[0]][1] = tree.children[ordered_items[0]]
		else:
			update_header(header_table[ordered_items[0]][1], 
				tree.children[ordered_items[0]])
	
	if len(ordered_items) > 1:
		update_tree(ordered_items[1::], tree.children[ordered_items[0]], 
			header_table, count)




def create_tree(data_dict, min_support=1.0):
	'''
		create a FP tree
	'''
	header_table = {}
	for trans in data_dict:
		for item in trans:
			#get the total count of each item
			header_table[item] = header_table.get(item, 0) + data_dict[trans]
	
	#remove items not meeting minimum support
	for item in list(header_table):
		if header_table[item]< min_support:
			del(header_table[item])

	frequent_set = set(header_table)
	if(len(frequent_set) == 0):
		return None, None
	for item in header_table:
		header_table[item] = [header_table[item], None]

	tree = TreeNode('Null Set', 1, None)
	for (trans, count) in data_dict.items():
		local_data = {}
		for item in trans:
			if item in frequent_set:
				local_data[item] = header_table[item][0]
		if(len(local_data) > 0):
			ordered_items = [v[0] for v in sorted(local_data.items(), 
				key = lambda p: p[1], reverse = True)]
			update_tree(ordered_items, tree, header_table, count)

	return tree, header_table

def ascent_tree(leaf_node, prefix_path):
	'''
		ascend the tree
	'''
	if leaf_node.parent_node != None:
		prefix_path.append(leaf_node.name)
		ascent_tree(leaf_node.parent_node, prefix_path)

def find_prefix_path(base_path, tree_node):
	'''
		find the prefix path
	'''
	conditional_bases = {}
	while(tree_node != None):
		prefix_path = []
		ascent_tree(tree_node, prefix_path)
		if (len(prefix_path) > 1):
			conditional_bases[frozenset(prefix_path[1:])] = tree_node.count
		tree_node = tree_node.node_link

	return conditional_bases

def mine_tree(tree, header_table, min_support, prefix, frequent_list):
	'''
		create conditional trees and prefix paths and conditional bases
	'''
	#sort header table
	ordered_items = [v[0] for v in sorted(header_table.items(), 
		key = lambda p: str(p[1]))]
	#start from bottom of header table
	for base_path in ordered_items:
		new_frequent_set = prefix.copy()
		new_frequent_set.add(base_path)
		frequent_list.append(new_frequent_set)
		conditional_bases = find_prefix_path(base_path, 
			header_table[base_path][1])
		conditional_tree, header = create_tree(conditional_bases, 
			min_support)
		if header != None:
			print("\nconditional tree for: ", new_frequent_set)
			conditional_tree.display(1)
			mine_tree(create_tree, header, min_support, 
				new_frequent_set, frequent_list)










































