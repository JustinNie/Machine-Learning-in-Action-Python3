#test the decision tree
#Author: Justin Nie
#Date: 2018/1/26

from numpy import *

from trees import *
from tree_plotter import *

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
lenses_tree = create_tree(lenses, lenses_labels)
store_tree(lenses_tree, 'lenses_tree_storage.txt')
lenses_tree = grab_tree('lenses_tree_storage.txt')

create_plot(lenses_tree)