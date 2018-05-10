import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import sample
from tensorflow.examples.tutorials.mnist import input_data

# Load and sample the dataset
mnist_images = input_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0).test.images
mnist_images = sample(list(mnist_images), 10)

# Create the variables for the neural network
X  = tf.placeholder(tf.float32, [None, 28, 28, 1])
W1 = tf.get_variable("W1", shape=[28*28, 30])
B1 = tf.get_variable("B1", shape=[30])
W2 = tf.get_variable("W2", shape=[30, 10])
B2 = tf.get_variable("B2", shape=[10])

# Create the neural network model from the created variables
# Also, create the training method used to optimize the network
# Using gradient descent to minize the square error of the network
XX = tf.reshape(X, [-1, 28*28])
H = tf.nn.sigmoid(tf.add(tf.matmul(XX, W1), B1))
Y = tf.nn.sigmoid(tf.add(tf.matmul(H, W2), B2))

# Initialize TensorFlow session
initializer = tf.global_variables_initializer()
session = tf.Session()
session.run(initializer)

# Restore trained model
path = "./models/sigmoid_nn/sigmoid_nn"
saver = tf.train.Saver()
saver.restore(session, path)

# Predict unseen digits
classifications = session.run(tf.argmax(Y, 1), feed_dict={X: mnist_images})
print("prediction: %s" % ''.join(str(digit) for digit in classifications))

# Plot the digits
plt.rc("image", cmap="gray")
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(mnist_images[i].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
plt.tight_layout()
plt.show()
