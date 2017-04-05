# _*_ coding: utf-8 _*_

"""

tensorflow_simple created by xiangkun on 2017/3/25

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)

# dataset
xx = np.random.randint(0,1000,[1000,3])/1000.
yy = xx[:,0] * 2 + xx[:,1] * 1.4 + xx[:,2] * 3

# yy = yy.reshape(yy.shape[0], 1)
print "xx,yy shape: {},{}".format(xx.shape, yy.shape)

# model
x = tf.placeholder(tf.float32, shape=[None, 3])
y_ = tf.placeholder(tf.float32, shape=[None])
W1 = weight_variable([3, 1])
y = tf.matmul(x, W1)

# training and cost function
# FIX: y to tf.squeeze(y)
cost_function = tf.reduce_mean(tf.square(tf.squeeze(y) - y_))
train_function = tf.train.AdamOptimizer(1e-2).minimize(cost_function)

# create a session
sess = tf.Session()

# train
# FIX: use tf.global_variable_initializer
sess.run(tf.global_variables_initializer())
for i in range(10000):
    sess.run(train_function, feed_dict={x:xx, y_:yy})
    if i % 1000 == 0:
        print(sess.run(cost_function, feed_dict={x:xx, y_:yy}))