import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = "layer" + str(n_layer)
    with tf.name_scope("layers"):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.histogram_summary(layer_name + "/weights", weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, "b")
            tf.histogram_summary(layer_name + "/biases", biases)
        with tf.name_scope("Wx_plus_b"):
            wx_plus_b = tf.matmul(inputs, weights) + biases

        if activation_function:
            outputs = activation_function(wx_plus_b)
        else:
            outputs = wx_plus_b
        tf.histogram_summary(layer_name + "/outputs", outputs)
        return outputs

# Make up some real data.
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, size=x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define place hold for inputs to network.
with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

l1 = add_layer(xs, 1, 10, n_layer=0, activation_function=tf.nn.relu)
# Linear regression, no activation or linear activation.
prediction = add_layer(l1, 10, 1, n_layer=1, activation_function=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.scalar_summary("loss", loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_step = optimizer.minimize(loss)

#
init = tf.initialize_all_variables()

sess = tf.Session()

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("./graphs/", sess.graph)


sess.run(init)
#
# plt.ion()
# figure, axes = plt.subplots(2, 1, sharex=False, sharey=False)
# top = axes[0]
# bot = axes[1]
# top.scatter(x_data, y_data, c='b')
#
# perc = 1
#
#
# def vis():
#     loss_val = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
#     prediction_val = sess.run(prediction, feed_dict={xs: x_data})
#     lines = top.plot(x_data, prediction_val, c='r')
#     bot.scatter(i, loss_val, c='k')
#     plt.pause(0.05)
#     top.lines.remove(lines[0])
#
for i in range(1001):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
else:
    # vis()
    pass
#
#
# plt.ioff()
# plt.show()
