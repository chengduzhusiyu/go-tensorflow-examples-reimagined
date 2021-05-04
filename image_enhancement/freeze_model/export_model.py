import os
import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.tools import freeze_graph
from model import SRGAN_g


# Uncomment the following line to print the GPU and tf and tl log
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.DEBUG)
# tl.logging.set_verbosity(tl.logging.DEBUG)

def preprocess(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x


def export_model():
    """Load the model in TensorLayer's way and save
    the frozen graph

    Args:
        None

    Returns:
        None
    """

    # create folders to save result images
    checkpoint_dir = "checkpoint"

    ###========================== DEFINE MODEL ============================###
    t_image = tf.placeholder('float32', [None, None, None, 3],
                             name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    tl.layers.initialize_global_variables(s