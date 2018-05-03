import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from tensorflow.examples.tutorials.mnist import input_data

# Load and shuffle the dataset
mnist_images = input_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0).train.images
shuffle(mnist_images)

# Plot some examples of the digits
plt.rc("image", cmap="gray")
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(mnist_images[i].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
plt.tight_layout()
plt.show()
