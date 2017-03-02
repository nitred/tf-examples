from pprint import pprint

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


def weight_variable(shape, parent_name=None, summary=False):
    weights = tf.get_variable(name="weights",
                              shape=shape,
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if summary is True:
        assert parent_name is not None, "`parent_name` should be give if `summary` is True."
        name = parent_name + "/weights"
        tf.summary.histogram(name=name, values=weights)
    return weights


def bias_variable(shape, parent_name=None, summary=False):
    biases = tf.get_variable(name="biases",
                             shape=shape,
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.0))
    if summary is True:
        assert parent_name is not None, "`parent_name` should be given if `summary` is True."
        name = parent_name + "/biases"
        tf.summary.histogram(name=name, values=biases)
    return biases


def fc_layer(inputs, in_size, out_size, activation_function, scope_name, summary=False):
    with tf.variable_scope(scope_name) as scope:
        W = weight_variable(shape=[in_size, out_size], parent_name=scope.name, summary=True)
        b = bias_variable(shape=[out_size], parent_name=scope.name, summary=True)
        if activation_function is "linear":
            outputs = tf.add(x=tf.matmul(inputs, W), y=b, name="linear")
        else:
            outputs = activation_function(tf.add(tf.matmul(inputs, W), b), name=activation_function.__name__)
        if summary is True:
            name = scope.name + "/outputs"
            tf.summary.histogram(name=name, values=outputs)
        return outputs


with tf.variable_scope("MLP") as scope:
    x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x_input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y_input")
    fc1 = fc_layer(inputs=x, in_size=1, out_size=8, activation_function=tf.nn.relu, scope_name="fc1", summary=True)
    fc2 = fc_layer(inputs=fc1, in_size=8, out_size=1, activation_function="linear", scope_name="fc2", summary=True)
    prediction = fc2


with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), axis=1))
    tf.summary.scalar(name="loss", tensor=loss)


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir="./logs/", graph=sess.graph, flush_secs=1)


def sample_x(n):
    return (20 * np.random.rand(n)[:, np.newaxis]) - 20  # [-20, 20]


def sample_y(x):
    noise = np.random.normal(loc=0.0, scale=0.05, size=x.shape[0]).reshape(x.shape)
    return 8 * x + 0.9


def plt_setup():
    plt.ion()
    pass


batch_size = 16
for i in range(2001):
    x_train = sample_x(batch_size)
    y_train = sample_y(x_train)
    sess.run(train_step, feed_dict={x: x_train, y: y_train})

    if i % 10 == 0:
        x_valid = sample_x(batch_size)
        y_valid = sample_y(x_valid)
        _loss = sess.run(loss, feed_dict={x: x_valid, y: y_valid})
        pprint(_loss)
        # plt.scatter(i, _loss, s=1, c='r')
        # plt.xlim(0, 2000)
        # plt.ylim(0, 10000)
        summary = sess.run(merged_summaries, feed_dict={x: x_valid, y: y_valid})
        writer.add_summary(summary=summary, global_step=i)
        plt.pause(0.5)

# plt.ioff()
# plt.show()
