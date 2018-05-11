import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
from random import sample
from tensorflow.examples.tutorials.mnist import input_data

def build_traditional_nn():
    # Create the variables for the neural network
    X  = tf.placeholder(tf.float32, [None, 28, 28, 1])
    W1 = tf.get_variable("W1", [28*28, 30])
    B1 = tf.get_variable("B1", [30])
    W2 = tf.get_variable("W2", [30, 10])
    B2 = tf.get_variable("B2", [10])

    # Create the initial neural network model from the created variables
    XX = tf.reshape(X, [-1, 28*28])
    H = tf.nn.sigmoid(tf.add(tf.matmul(XX, W1), B1))

    return X, W1, B1, W2, B2, XX, H

def build_sigmoid_nn():
    # Create the sigmoid neural network model from the initial variables
    X, W1, B1, W2, B2, XX, H = build_traditional_nn()
    Y = tf.nn.sigmoid(tf.add(tf.matmul(H, W2), B2))

    return X, W1, B1, W2, B2, XX, H, Y

def build_softmax_nn():
    # Create the softmax neural network model from the initial variables
    X, W1, B1, W2, B2, XX, H = build_traditional_nn()
    Y = tf.nn.softmax(tf.add(tf.matmul(H, W2), B2))

    return X, W1, B1, W2, B2, XX, H, Y

def build_convolutional_nn():
    # Create the variables for the neural network
    X  = tf.placeholder(tf.float32, [None, 28, 28, 1])
    W1 = tf.get_variable("W1", [5, 5, 1, 4])
    B1 = tf.get_variable("B1", [4])
    W2 = tf.get_variable("W2", [5, 5, 4, 8])
    B2 = tf.get_variable("B2", [8])
    W3 = tf.get_variable("W3", [4, 4, 8, 12])
    B3 = tf.get_variable("B3", [12])
    W4 = tf.get_variable("W4", [7*7*12, 200])
    B4 = tf.get_variable("B4", [200])
    W5 = tf.get_variable("W5", [200, 10])
    B5 = tf.get_variable("B5", [10])

    # Create the convolutional neural network model from the created variables   
    C1 = tf.nn.relu(tf.add(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME'), B1))
    C2 = tf.nn.relu(tf.add(tf.nn.conv2d(C1, W2, strides=[1, 2, 2, 1], padding='SAME'), B2))
    C3 = tf.nn.relu(tf.add(tf.nn.conv2d(C2, W3, strides=[1, 2, 2, 1], padding='SAME'), B3))
    CC = tf.reshape(C3, [-1, 7*7*12])
    H = tf.nn.relu(tf.add(tf.matmul(CC, W4), B4))
    Y = tf.nn.softmax(tf.add(tf.matmul(H, W5), B5))

    return X, W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, C1, C2, C3, CC, H, Y

# Load argument
model = sys.argv[1]

# Load and sample the dataset
mnist_images = input_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0).test.images
mnist_images = sample(list(mnist_images), 10)

# Create the neural network model
if model == "sigmoid":
    X, W1, B1, W2, B2, XX, H, Y = build_sigmoid_nn()
elif model == "softmax":
    X, W1, B1, W2, B2, XX, H, Y = build_softmax_nn()
elif model == 'convolutional':
    X, W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, C1, C2, C3, CC, H, Y = build_convolutional_nn()
else:
    print("Invalid model!")
    exit()

# Initialize TensorFlow session
initializer = tf.global_variables_initializer()
session = tf.Session()
session.run(initializer)

# Restore trained model
path = "./models/%s_nn/%s_nn" % (model, model)
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
