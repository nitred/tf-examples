import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function:
        outputs = activation_function(wx_plus_b)
    else:
        outputs = wx_plus_b
    return outputs


# Make up some real data.
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, size=x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# plt.scatter(x_data, y_data, c='b')
# plt.show()

# define place hold for inputs to network.
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)  # Linear regression, no activation or linear activation.

#
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_step = optimizer.minimize(loss)

#
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

plt.ion()
figure, axes = plt.subplots(2, 1, sharex=False, sharey=False)
top = axes[0]
bot = axes[1]
top.scatter(x_data, y_data, c='b')

perc = 1


def vis():
    loss_val = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
    prediction_val = sess.run(prediction, feed_dict={xs: x_data})
    lines = top.plot(x_data, prediction_val, c='r')
    bot.scatter(i, loss_val, c='k')
    plt.pause(0.05)
    top.lines.remove(lines[0])

for i in range(4001):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % perc == 0:
        perc = 2 * perc
        vis()
else:
    vis()


plt.ioff()
plt.show()
