# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# NOTICE: There may be some modifications to this codebase my me (nitred)
# from the original code base. This file is used by me for practice and revision.
# This file has not been copied as is, but typed out.

from __future__ import absolute_import, division, print_function

import os
import re
import sys
import tarfile

import cifar10_input
import tensorflow as tf
from six.moves import urllib

FLAGS = tf.app.flags.FLAGS

# Basic model
tf.app.flags.DEFINE_integer(flag_name='batch_size', default_value=128, docstring="batch_size")
tf.app.flags.DEFINE_string(flag_name='data_dir',
                           default_value="/media/nitred/mydata/datasets/cifar-10/",
                           docstring="data_dir")
tf.app.flags.DEFINE_boolean(flag_name='use_fp16', default_value=False, docstring="Train the model using fp16.")

# Global constraints describing the cifar-10 dataset
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INTIAL_LEARNING_RATE = 0.1


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tars'


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor

    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd=None):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    # What's happening here???
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir.
    """
    if not FLAGS.data_dir:
        raise ValueError("Please supply a data_dir")
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError("Please supply a data_dir.")
    data_dir = os.path.join(FLAGS.data_dir, "cifar-10-batches-bin")
    images, labels = cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images):
    """Build the CIFAR-10 model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Returns:
        Logits.
    """
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(name='weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(input=images, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(name='biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(value=conv, bias=biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(x=conv1)
    pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
    norm1 = tf.nn.local_response_normalization(input=pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(name='weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(input=norm1, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(name='biases', shape=[64], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(value=conv, bias=biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(x=conv2)
    norm2 = tf.nn.local_response_normalization(input=conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.nn.max_pool(value=norm2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[FLAGS.batch_size, -1])  # [-1, 24 * 24 * 64]
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay(name='weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(name='biases', shape=[384], initializer=tf.constant_initializer(value=0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(x=local3)

    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay(name='weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(name='biases', shape=[192], initializer=tf.constant_initializer(value=0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(x=local4)

    with tf.variable_scope('local5') as scope:
        weights = _variable_with_weight_decay(name='weights', shape=[192, NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu(name='biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        # NOTE: Using tf.add instead of '+' so that we can use the `name` parameter.
        # NOTE: We're going to do cross_entropy. The cross_entropy func provided by tensorflow already does softmax
        # and expects a linear input instead. So we don't do softmax here, instead just do a linear activation.
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(x=softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".

    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9, name="avg")
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(name=l.op.name + " raw ", tensor=1)
        tf.summary.scalar(name=l.op.name, tensor=loss_averages.average(1))  # ???

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.

    Returns:
        train_op: op for training.
    """
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(learning_rate=INTIAL_LEARNING_RATE,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar(name='learning_rate', tensor=lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # NOTE: Why are we doing this instead of opt.minimize(loss)?
    # Compute Gradients
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    # Apply Gradients.
    apply_gradient_op = opt.apply_gradients(grads_and_vars=grads, global_step=global_step)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
