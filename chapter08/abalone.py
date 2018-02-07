#Predict the age of a shellfish called abalone
#Authoor: Justin Nie
#Date: 2018/2/7

from numpy import *
from regression import *
import matplotlib.pyplot as plt

abalone_x, abalone_y = load_dataset('abalone.txt')
y_hat1 = lwlr_test(abalone_x[0: 99], abalone_x[0: 99], abalone_y[0: 99], 0.1)
y_hat2 = lwlr_test(abalone_x[0: 99], abalone_x[0: 99], abalone_y[0: 99], 1.0)
y_hat3 = lwlr_test(abalone_x[0: 99], abalone_x[0: 99], abalone_y[0: 99], 10.)

rss_error1 = rss_error(abalone_y[0: 99], y_hat1.T)
print(rss_error1)
rss_error2 = rss_error(abalone_y[0: 99], y_hat2.T)
print(rss_error2)
rss_error3 = rss_error(abalone_y[0: 99], y_hat3.T)
print(rss_error3)


y_hat_new1 = lwlr_test(abalone_x[100: 199], abalone_x[0: 99], abalone_y[0: 99], 0.1)
y_hat_new2 = lwlr_test(abalone_x[100: 199], abalone_x[0: 99], abalone_y[0: 99], 1.0)
y_hat_new3 = lwlr_test(abalone_x[100: 199], abalone_x[0: 99], abalone_y[0: 99], 10.)

rss_error_new1 = rss_error(abalone_y[100: 199], y_hat_new1.T)
print(rss_error_new1)
rss_error_new2 = rss_error(abalone_y[100: 199], y_hat_new2.T)
print(rss_error_new2)
rss_error_new3 = rss_error(abalone_y[100: 199], y_hat_new3.T)
print(rss_error_new3)


ws, cache = standard_regression(abalone_x[0: 99], abalone_y[0: 99])
y_hat = mat(abalone_x[100: 199]) * ws
rss_error = rss_error(abalone_y[100: 199], y_hat.T)
print(rss_error)


ridge_weights = ridge_test(abalone_x, abalone_y)
print(ridge_weights)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridge_weights)
plt.show()


stage_wise_weights = stage_wise(abalone_x, abalone_y, 0.001, 5000)
print(stage_wise_weights)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(stage_wise_weights)
plt.show()


















