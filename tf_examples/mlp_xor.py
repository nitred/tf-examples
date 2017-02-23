import numpy as np
import tensorflow as tf


def sample_x(n):
    x = np.empty([n, 2], dtype=np.int)
    x[:, 0] = np.random.binomial(n=1, p=0.5, size=n)
    x[:, 1] = np.random.binomial(n=1, p=0.5, size=n)
    return x.astype(np.float32)


def sample_y(x):
    y = np.logical_xor(x[:, 0], x[:, 1])[:, np.newaxis]
    return y


def weight_variable(shape):
    return tf.get_variable("weights",
                           shape=shape,
                           dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))


def bias_variable(shape):
    return tf.get_variable("biases",
                           shape=shape,
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(value=0.1))


def add_layer(inputs, in_size, out_size, activation_function, scope_name):
    with tf.variable_scope(scope_name):
        W = weight_variable([in_size, out_size])
        b = bias_variable([out_size])
        if activation_function == "linear":
            output = tf.matmul(inputs, W) + b
        else:
            output = activation_function(tf.matmul(inputs, W) + b)
        return output


def network(x):
    l1 = add_layer(x, 2, 8, tf.nn.tanh, "l1")
    l2 = add_layer(l1, 8, 1, tf.nn.sigmoid, "l2")
    return l2


with tf.variable_scope("XOR"):
    x = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 1])
    XOR = network(x)


loss_XOR = tf.reduce_mean(tf.square(XOR - y))

learning_rate = tf.train.exponential_decay(decay_rate=0.8, decay_steps=100,
                                           global_step=tf.Variable(0), learning_rate=0.5)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

vars_XOR = tf.get_collection(tf.GraphKeys.)
opt_XOR = optimizer.minimize(loss_XOR, var_list=vars_XOR)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


epochs = 10000
batch_size = 16
for i in range(epochs):
    x_train = sample_x(batch_size)
    y_train = sample_y(x_train)
    sess.run(opt_XOR, feed_dict={x: x_train, y: y_train})


x_test = sample_x(10)
y_test = sample_y(x_test)
y_pred = sess.run(XOR, feed_dict={x: x_test})

print("y_test")
print(y_test)

print("y_pred")
print(y_pred)
