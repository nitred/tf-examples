"""Contains code basic implementation of GAN modeling a Gaussian.

- This is the basic version which is imperfect.
- The next versions in another file will aim to improve this model.
- Learning rate decay doesn't seem to work.
"""
from pprint import pprint

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns


def sample_z(n):
    return np.random.rand(n)[:, np.newaxis]


def sample_x(n):
    return np.random.normal(loc=-2, scale=1.0, size=n)[:, np.newaxis]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.get_variable("weights",
                           shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))


def bias_variable(shape):
    return tf.get_variable("biases",
                           shape=shape,
                           initializer=tf.constant_initializer(0.0))


def add_layer(inputs, in_size, out_size, activation_function, scope_name):
    with tf.variable_scope(scope_name):
        w_fc = weight_variable([in_size, out_size])
        b_fc = bias_variable([out_size])
        if activation_function == "linear":
            h_fc = tf.matmul(inputs, w_fc) + b_fc
        else:
            h_fc = activation_function(tf.matmul(inputs, w_fc) + b_fc)
        return h_fc


def generator(z):
    g0 = add_layer(z, 1, 4, tf.nn.softplus, "layer0")
    g1 = add_layer(g0, 4, 1, "linear", "layer1")
    return g1


def discriminator(x):
    d0 = add_layer(x, 1, 8, tf.nn.tanh, "layer0")
    d1 = add_layer(d0, 8, 8, tf.nn.tanh, "layer1")
    d2 = add_layer(d1, 8, 1, tf.nn.sigmoid, "layer2")
    return d2


with tf.variable_scope("D_pre") as scope:
    x_pre = tf.placeholder(tf.float32, [None, 1])  # samples drawn at random from normal
    y_pre = tf.placeholder(tf.float32, [None, 1])  # probability from normal.pdf(x_pre)
    D_pre = discriminator(x_pre)


with tf.variable_scope("G"):
    z = tf.placeholder(tf.float32, [None, 1])
    G = generator(z)


with tf.variable_scope("D") as scope:
    x = tf.placeholder(tf.float32, [None, 1])
    D1 = discriminator(x)
    scope.reuse_variables()
    D2 = discriminator(G)

loss_d_pre = tf.reduce_mean(tf.square(D_pre - y_pre))
loss_g = tf.reduce_mean(-tf.log(D2))
loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))

vars_d_pre = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D_pre")
vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")
vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

learning_rate = tf.train.exponential_decay(
    learning_rate=0.5, global_step=tf.Variable(0), decay_steps=500, decay_rate=0.8)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
opt_d_pre = optimizer.minimize(loss_d_pre, var_list=vars_d_pre)
opt_g = optimizer.minimize(loss_g, var_list=vars_g)
opt_d = optimizer.minimize(loss_d, var_list=vars_d)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()
fig, ax = plt.subplots()
plt.title("Gaussian GAN")
plt.xlabel("X Samples")
plt.ylabel("Probability")
epochs_pre_train = 101
epochs_train = 10001
batch_size = 32


def plot_histogram_outline(data, c='r'):
    ax.set_xlim([-8, 8])
    ax.set_ylim([-0.1, 1.0])
    try:
        ax.lines[1].remove()  # clear lines, don't clear the first true distribution
    except Exception:
        pass
    sns.distplot(data, hist=False, ax=ax, color=c)
    plt.pause(0.1)


def test():
    # test plot
    z_test = sample_z(1000)
    g_test = sess.run(G, feed_dict={z: z_test})
    loss_g_test = sess.run(loss_g, feed_dict={z: z_test})
    print("G_loss: {}".format(loss_g_test))
    if not np.isnan(g_test[0][0]):
        plot_histogram_outline(g_test)
    else:
        print("nan {}".format(i))


# plot known underlying true distribution of x
x_true = sample_x(10000)
plot_histogram_outline(x_true, c='b')

# pre train
print("Pre-Training")
for i in range(epochs_pre_train):
    x_pre_train = sample_x(batch_size)
    y_pre_train = sp.stats.norm.pdf(x_pre_train)
    sess.run([loss_d_pre, opt_d_pre], feed_dict={x_pre: x_pre_train, y_pre: y_pre_train})
    if i % 50 == 0:
        print(sess.run(loss_d_pre, feed_dict={x_pre: x_pre_train, y_pre: y_pre_train}))
else:
    # assign weights pre-trained D_pre to D1 and D2
    vars_d_pre_trained = sess.run(vars_d_pre)
    for var_d, var_d_pre_trained in zip(vars_d, vars_d_pre_trained):
        sess.run(var_d.assign(var_d_pre_trained))

print("Training")
for i in range(epochs_train):
    # train discriminator
    for _ in range(1):
        x_train = sample_x(batch_size)
        z_train = sample_z(batch_size)
        sess.run([loss_d, opt_d], feed_dict={x: x_train, z: z_train})
    # train generator
    z_train = sample_z(batch_size)
    sess.run([opt_g], feed_dict={z: z_train})

    if i % 200 == 0:
        print("Epoch: {}".format(i))
        test()


print("Done")
plt.ioff()
plt.show()
