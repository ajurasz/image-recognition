from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import data_utils
import nn_classifier

TRAIN_DIR = './logs'
IMAGE_PIXELS = 3072
CLASSES = len(data_utils.CLASSES)
HIDDEN_1 = 120
REG_CONSTANT = 0.1

images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
global_step = tf.Variable(0, name="global_step", trainable=False)
logits = nn_classifier.inference(images_placeholder, IMAGE_PIXELS,
                                 HIDDEN_1, CLASSES, reg_constant=REG_CONSTANT)

saver = tf.train.Saver()


def is_cat_image(image):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring variables from checkpoint')
            saver.restore(sess, ckpt.model_checkpoint_path)
            current_step = tf.train.global_step(sess, global_step)
            print('Current step: {}'.format(current_step))

            prediction = tf.argmax(logits, 1)
            best = sess.run([prediction], feed_dict={images_placeholder: data_utils.get_image_pixels(image, IMAGE_PIXELS)})
            print(best)
            return best[0][0] == 3
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_file',
        type=str,
        required=True,
        help='Absolute path to image file.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print("Is this a cat image: {}".format(is_cat_image(FLAGS.image_file)))
