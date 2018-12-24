import tensorflow as tf
import numpy as np
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm

from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common import set_global_seeds

seed = None


def nature_cnn_lite(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    # TFLite max filter size: 5, max stride: 2, reducing first layer from 8/4 to 5/2
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=5, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class NatureLitePolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        global seed
        # we need to set seed here, once we are in the graph
        if seed != None:
            set_global_seeds(seed)
        super(NatureLitePolicy, self).__init__(*args, cnn_extractor=nature_cnn_lite,
            **kwargs)

def set_seed(new_seed=None):
    global seed
    seed = new_seed
    set_global_seeds(new_seed)