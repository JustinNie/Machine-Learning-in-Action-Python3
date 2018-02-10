#Test Apriori algorithm.
#Author: Justin Nie
#Date: 2018/2/10

from numpy import *
from apriori import *

dataset = load_simdata()
frequent, support_data = apriori(dataset)
#print(frequent)
#print(support_data)
rules = get_rules(frequent, support_data)
print(rules)


