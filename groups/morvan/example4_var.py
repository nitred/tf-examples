# Variables, assign and update.

import tensorflow as tf

state = tf.Variable(0, name='counter')
# print(state.name)

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # assign new value to state

init = tf.initialize_all_variables()  # must have this if you have variables.

with tf.Session() as sess:
    sess.run(init)
    for _ in range(20):
        sess.run(update)
        print(state.name, sess.run(state))
