
import os
import tensorflow as tf
import tensorflow.contrib.resampler

export_path = "/home/abduld/mlperf/inference/v0.5/translation/gnmt/tensorflow/savedmodel"

with tf.Session(graph=tf.Graph()) as sess:
    tf.save