from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import data_utils

LEARNING_RATE = 0.005
MAX_STEPS = 5000
BATCH_SIZE = 100

# Load dataset
data_set = data_utils.load_data()

# TF input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])  # None = any value, 32 X 32 x 3 = 3072
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# TF variables
weights = tf.Variable(tf.zeros([3072, 10]))
biases = tf.Variable(tf.zeros([10]))

# TF classifier result
logits = tf.matmul(images_placeholder, weights) + biases

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels_placeholder
))

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# Prediction vs true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Accuracy of prediction
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run
with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    for i in range(MAX_STEPS):
        indices = np.random.choice(data_set['images_train'].shape[0], BATCH_SIZE)
        images_batch = data_set['images_train'][indices]
        labels_batch = data_set['labels_train'][indices]

        # Print accuracy after each parameter update
        if i % BATCH_SIZE == 0:
            train_accuracy = session.run(accuracy, feed_dict={
                images_placeholder: images_batch, labels_placeholder: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

        session.run(train_step, feed_dict={images_placeholder: images_batch,
                                    labels_placeholder: labels_batch})

    # Evaluate test set after training model
    test_accuracy = session.run(accuracy, feed_dict={
        images_placeholder: data_set['images_test'],
        labels_placeholder: data_set['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))