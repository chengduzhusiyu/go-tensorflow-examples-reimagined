import os
import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.tools import freeze_graph
from model import SRGAN_g


# Uncomment the following line to print the GPU and tf and tl log
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.DEBUG)
# tl.logging.set_ve