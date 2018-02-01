#create a decision tree based on a dictionary
#Author: Justin Nie
#Date: 2018/1/27

import matplotlib.pyplot as plt


decision_node = dict(boxstyle = "sawtooth", fc = "0.8")
leaf_node = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = '<-')

def get_number_leafs(tree):
	'''get the number of the leaf node'''
	number_leafs = 0
	first_sides = list(tree.keys())
	first_str = first_sides[0]
	second_dict = tree[first_str]

	for key in second_dict.keys():
		if(type(second_dict[key]).__name__ == 'dict'):
			number_leafs += get_number_leafs(second_dict[key])
		else:
			number_leafs += 1
	return number_leafs

def get_tree_depth(tree):
	'''get the depth of the tree'''
	max_depth = 0
	first_sides = list(tree.keys())
	first_str = first_sides[0]	
	second_dict = tree[first_str]

	for key in second_dict.keys():
		if(type(second_dict[key]).__name__ == 'dict'):
			depth = get_tree_depth(second_dict[key]) +1
		else:
			depth = 1
		if depth > max_depth:
			max_depth = depth

	return max_depth

def plot_mid_text(center_plot, parent_plot, text_string):
	'''plot text string between center_plot and parent_plot'''
	x_mid = (parent_plot[0] - center_plot[0]) / 2.0 + center_plot[0]
	y_mid = (parent_plot[1] - center_plot[1]) / 2.0 + center_plot[1]
	create_plot.ax1.text(x_mid, y_mid, text_string)

def plot_node(node_text, center_plot, parent_plot, node_type):
	'''
		plot a node
		node_text: the text to describe the node
		center_plot: the node center coordinates
		parent_plot: parent node center coordinates
		node_type: the node fashion
	'''
	create_plot.ax1.annotate(node_text, xy = parent_plot, 
		xycoords = 'axes fraction', xytext = center_plot, 
		textcoords = 'axes fraction', va = 'center', ha = 'center', 
		bbox = node_type, arrowprops = arrow_args)

def plot_tree(tree, parent_plot, node_text):
	'''
		plot the decision tree based on the tree
	'''
	number_leafs = get_number_leafs(tree)
	depth = get_tree_depth(tree)
	first_sides = list(tree.keys())
	first_str = first_sides[0]	

	center_plot = (plot_tree.x_off + (1.0 + float(number_leafs))/2.0
		/plot_tree.total_width, plot_tree.y_off)
	plot_mid_text(center_plot, parent_plot, node_text)
	plot_node(first_str, center_plot, parent_plot, decision_node)
	
	second_dict = tree[first_str]
	plot_tree.y_off = plot_tree.y_off - 1.0/plot_tree.total_depth

	for key in second_dict.keys():
		if type(second_dict[key]).__name__ == 'dict':
			plot_tree(second_dict[key], center_plot, str(key))
		else:
			plot_tree.x_off = plot_tree.x_off + 1.0/plot_tree.total_width
			plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), 
				center_plot, leaf_node)
			plot_mid_text((plot_tree.x_off, plot_tree.y_off), 
				center_plot, str(key))
			
	plot_tree.y_off = plot_tree.y_off + 1.0/plot_tree.total_depth


def create_plot(tree):
	'''
		create the decision tree
	'''
	fig = plt.figure(1, facecolor = 'white')
	fig.clf()
	axprops = dict(xticks = [], yticks = [])
	create_plot.ax1 = plt.subplot(111, frameon = False, **axprops)
	plot_tree.total_width = float(get_number_leafs(tree))
	plot_tree.total_depth = float(get_tree_depth(tree))
	plot_tree.x_off = -0.5/plot_tree.total_width
	plot_tree.y_off = 1.0
	plot_tree(tree, (0.5, 1.0), '')
	plt.show()






















