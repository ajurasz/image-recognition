import numpy as np
import tensorflow as tf


def inference(images, image_pixels, hidden_units, classes, reg_constant=0):
    with tf.variable_scope('Layer1'):
        weights = tf.get_variable(
            name='weights',
            shape=[image_pixels, hidden_units],
            initializer=tf.truncated_normal_initializer(
                stddev=1.0 / np.sqrt(float(image_pixels))),
            regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
        )

        biases = tf.Variable(tf.zeros([hidden_units]), name='biases')

        # Output
        hidden = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.variable_scope('Layer2'):
        weights = tf.get_variable('weights', [hidden_units, classes],
          initializer=tf.truncated_normal_initializer(
            stddev=1.0 / np.sqrt(float(hidden_units))),
          regularizer=tf.contrib.layers.l2_regularizer(reg_constant))

        biases = tf.Variable(tf.zeros([classes]), name='biases')

        # Output
        logits = tf.matmul(hidden, weights) + biases

        # # Define summery-operation for 'logits'-variable
        # tf.summary.histogram('logits', logits)

    return logits


def loss(logits, labels):
    with tf.name_scope('Loss'):
        # Operation to determine the cross entropy between logits and labels
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy'))

        # Operation for the loss function
        loss = cross_entropy + tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))

        # # Add a scalar summary for the loss
        # tf.summary.scalar('loss', loss)

    return loss


def training(loss, learning_rate):
    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a gradient descent optimizer (which also increments the global step counter)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    return train_step


def evaluation(logits, labels):
    with tf.name_scope('Accuracy'):
        # Operation comparing prediction with true label
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)

        # Operation calculating the accuracy of the predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # # Summary operation for the accuracy
        # tf.summary.scalar('train_accuracy', accuracy)

    return accuracy

