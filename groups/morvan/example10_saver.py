import tensorflow as tf

# # SAVING
# # remember to define the same dtype and shape when restoring.
# # The saver can only save the weights (variables).
# W = tf.Variable([[1, 2, 3], [1, 2, 3]],
#                 dtype=tf.float32,
#                 name='weights')
# b = tf.Variable([[1, 2, 3]],
#                 dtype=tf.float32,
#                 name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "myNN/save_net.ckpt")
#     print("Save to path:", save_path)


# RESTORING
# restore variables
# redefine the same shape and dtype.
W = tf.Variable(tf.zeros([2, 3]), dtype=tf.float32, name='weights')
b = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32, name='biases')
# NO INIT NEEDED

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./myNN/save_net.ckpt")
    print('weights:', sess.run(W))
    print('biases:', sess.run(b))
