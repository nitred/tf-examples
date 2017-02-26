import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def add_layer(inputs, in_size, out_size, layer_number, activation_function=None):
    weights = tf.Variable(tf.random_normal(shape=[in_size, out_size]))
    biases = tf.Variable(tf.zeros(shape=[1, out_size]))
    Wx_plus_b = tf.matmul(inputs, weights) + biases  # inputs*weights vs weights*inputs
    if activation_function:
        outputs = activation_function(Wx_plus_b)
    else:
        outputs = Wx_plus_b
    return outputs

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# placeholders
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28 = 784
ys = tf.placeholder(tf.float32, [None, 10])  # 10 classes.

# add output layer.
prediction = add_layer(xs, in_size=784, out_size=10, layer_number=0, activation_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.6)
train_step = optimizer.minimize(cross_entropy)

# session
sess = tf.Session()
sess.run(tf.initialize_all_variables())


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pred = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
