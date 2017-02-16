from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


def sample_z(n):
    return np.random.rand(n)[:, np.newaxis]


def sample_x(n):
    return np.random.normal(loc=-1, scale=1.0, size=n)[:, np.newaxis]


# placeholders
z = tf.placeholder(tf.float32, [None, 1])
x = tf.placeholder(tf.float32, [None, 1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable("weights",
                           shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))


def bias_variable(shape):
    return tf.get_variable("biases",
                           shape=shape,
                           initializer=tf.constant_initializer(0.1))


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
    g_fc1 = add_layer(z, 1, 16, tf.nn.softplus, "layer1")
    g_fc2 = add_layer(g_fc1, 16, 16, tf.nn.softplus, "layer2")
    g_fc3 = add_layer(g_fc2, 16, 1, "linear", "layer3")
    return g_fc3


def discriminator(x):
    d = add_layer(x, 1, 16, tf.nn.tanh, "layer1")
    d = add_layer(d, 16, 16, tf.nn.tanh, "layer2")
    d = add_layer(d, 16, 1, tf.nn.sigmoid, "layer3")
    return d


with tf.variable_scope("G"):
    G = generator(z)


with tf.variable_scope("D") as scope:
    D1 = discriminator(x)
    scope.reuse_variables()
    D2 = discriminator(G)

loss_g = tf.reduce_mean(-tf.log(D2))
loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

vars = tf.trainable_variables()
vars_g = [var for var in vars if var.name.startswith('G/')]
vars_d = [var for var in vars if var.name.startswith('D/')]

opt_g = optimizer.minimize(loss_g, var_list=vars_g)
opt_d = optimizer.minimize(loss_d, var_list=vars_d)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# z_test = sample_z(1)
# print("z", sess.run(z, feed_dict={z: z_test}))
# print("G", sess.run(G, feed_dict={z: z_test}))
# print("D2", sess.run(D2, feed_dict={z: z_test}))
# print("train", sess.run(G, feed_dict={z: z_test}))


plt.ion()
fig, ax = plt.subplots()


def plot_histogram_outline(data, bins=40, c='r', keep=False):
    ax.set_xlim([-10, 10])
    ax.set_ylim([-0.1, 1])
    try:
        ax.lines[1].remove()  # clear lines
    except Exception:
        pass
    sns.distplot(data, hist=False, ax=ax, color=c)
    plt.pause(0.1)


x_test = sorted(sample_x(1000))
plot_histogram_outline(x_test, bins=100, c='b', keep=True)


epochs = 5000
batch_size = 64
for i in range(epochs):
    # train discriminator
    for _ in range(1):
        x_train = sample_x(batch_size)
        z_train = sample_z(batch_size)
        sess.run([loss_d, opt_d], feed_dict={x: x_train, z: z_train})
    # train generator
    z_train = sample_z(batch_size)
    sess.run([loss_g, opt_g], feed_dict={z: z_train})

    if i % 100 == 0:
        # test plot
        z_test = sample_z(1000)
        g_test = sorted(sess.run(G, feed_dict={z: z_test}))
        if not np.isnan(g_test[0][0]):
            # pprint(z_test)
            # pprint(g_test)
            # print(g_test[0][0])
            print(np.std(g_test))
            plot_histogram_outline([i[0] for i in g_test])
        else:
            print("nan {}".format(i))

print("Done")
plt.ioff()
plt.show()
