from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime
import os.path
import data_utils
import nn_classifier

LEARNING_RATE = 0.001
MAX_STEP = 2000
HIDDEN_1 = 120
BATCH_SIZE = 400
TRAIN_DIR = './logs'
REG_CONSTANT = 0.1
IMAGE_PIXELS = 3072
CLASSES = len(data_utils.CLASSES)

data_sets = data_utils.load_data()

images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

logits = nn_classifier.inference(images_placeholder, IMAGE_PIXELS,
                                 HIDDEN_1, CLASSES, reg_constant=REG_CONSTANT)

loss = nn_classifier.loss(logits, labels_placeholder)

train_step = nn_classifier.training(loss, LEARNING_RATE)

accuracy = nn_classifier.evaluation(logits, labels_placeholder)

# Define saver to save model state at checkpoints
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
    batches = data_utils.gen_batch(list(zipped_data), BATCH_SIZE, MAX_STEP)

    for i in range(MAX_STEP):
        batch = next(batches)
        images_batch, labels_batch = zip(*batch)
        feed_dict = {
            images_placeholder: images_batch,
            labels_placeholder: labels_batch
        }

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))

        # Perform a single training step
        sess.run([train_step, loss], feed_dict=feed_dict)

        # Periodically save checkpoint
        if (i + 1) % 1000 == 0:
            checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)
            print('Saved checkpoint')

    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: data_sets['images_test'],
        labels_placeholder: data_sets['labels_test']
    })
    print('Test accuracy {:g}'.format(test_accuracy))
