import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def sample_x(n):
    return np.random.rand(n).astype(np.float32)[:, np.newaxis]


def sample_y(x):
    y = (x * 0.1) + 0.8
    noise = 0.01 * np.random.rand()
    return y + noise


def weight_variable(shape):
    return tf.get_variable(name="weights",
                           shape=shape,
                           dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))


def bias_variable(shape):
    return tf.get_variable(name="biases",
                           shape=shape,
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(value=0.0))


x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W = weight_variable(shape=[1, 1])
b = bias_variable(shape=[1])

prediction = tf.matmul(x, W) + b

loss = tf.reduce_mean(tf.square(y - prediction))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
optimization = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


batch_size = 1

for i in range(1000):
    x_train = sample_x(batch_size)
    y_train = sample_y(x_train)
    sess.run(optimization, feed_dict={x: x_train, y: y_train})

print("y = Wx + b")
print("y = {}x + {}".format(*sess.run([W, b])))
