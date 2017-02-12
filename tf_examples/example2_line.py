# Simple line approximation
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

np.random.seed(1)
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Structure
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases


def predy(w, b):
    return lambda x: w * x + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss=loss)

init = tf.initialize_all_variables()  # VERY IMPORTANT

# End
sess = tf.Session()
sess.run(init)

plt.ion()

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        w = sess.run(weights)
        b = sess.run(biases)
        y_pred = predy(w, b)(x_data)
        print(step, w, b)

        plt.clf()
        plt.scatter(x_data, y_data, c='b')
        plt.scatter(x_data, y_pred, c='r')
        plt.xlim(-1.5, 1.5)
        plt.ylim(0, 1)
        plt.pause(0.2)
