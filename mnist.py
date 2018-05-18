import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Disable warnings
tf.logging.set_verbosity(tf.logging.ERROR)

def read_dataset():
    return input_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
