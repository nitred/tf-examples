import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def add_layer(inputs, in_size, out_size, layer_number, keep_prob, activation_function=None):
    layer_name = "layer" + str(layer_number)
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=keep_prob)
    if activation_function:
        outputs = activation_function(Wx_plus_b)
    else:
        outputs = Wx_plus_b
    tf.histogram_summary(layer_name + "/outputs", outputs)
    return outputs

# placeholder
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, shape=[None, 64])  # 8x8
ys = tf.placeholder(tf.float32, shape=[None, 10])

# add layers
l1 = add_layer(xs, 64, 50, layer_number=0, keep_prob=keep_prob, activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, layer_number=1, keep_prob=keep_prob, activation_function=tf.nn.softmax)


# train
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
tf.scalar_summary('loss-cross-entropy', cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_step = optimizer.minimize(cross_entropy)

# session
sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./logs/train/", sess.graph)
test_writer = tf.summary.FileWriter("./logs/test", sess.graph)
sess.run(tf.global_variables_initializer())


def compute_accuracy(x_test, y_test):
    global prediction
    y_pred = sess.run(prediction, feed_dict={xs: x_test, keep_prob: 1.0})
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: x_test, ys: y_test, keep_prob: 1.0})
    return result


for i in range(500):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        # compute_accuracy(X_test, y_test)
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1.0})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1.0})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
        print(compute_accuracy(X_test, y_test))
