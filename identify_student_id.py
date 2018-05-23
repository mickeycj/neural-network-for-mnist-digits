import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
from random import sample

def build_input_layer():
    return tf.placeholder(tf.float32, [None, 28, 28, 1])

def build_traditional_nn():
    # Create the variables for the neural network
    X  = build_input_layer()
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
    X  = build_input_layer()
    W1 = tf.get_variable("W1", [6, 6, 1, 6])
    B1 = tf.get_variable("B1", [6])
    W2 = tf.get_variable("W2", [5, 5, 6, 12])
    B2 = tf.get_variable("B2", [12])
    W3 = tf.get_variable("W3", [4, 4, 12, 24])
    B3 = tf.get_variable("B3", [24])
    W4 = tf.get_variable("W4", [7*7*24, 200])
    B4 = tf.get_variable("B4", [200])
    W5 = tf.get_variable("W5", [200, 10])
    B5 = tf.get_variable("B5", [10])

    # Create the convolutional neural network model from the created variables   
    C1 = tf.nn.relu(tf.add(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME'), B1))
    C2 = tf.nn.relu(tf.add(tf.nn.conv2d(C1, W2, strides=[1, 2, 2, 1], padding='SAME'), B2))
    C3 = tf.nn.relu(tf.add(tf.nn.conv2d(C2, W3, strides=[1, 2, 2, 1], padding='SAME'), B3))
    CC = tf.reshape(C3, [-1, 7*7*24])
    H = tf.nn.relu(tf.add(tf.matmul(CC, W4), B4))
    Y = tf.nn.softmax(tf.add(tf.matmul(H, W5), B5))

    return X, W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, C1, C2, C3, CC, H, Y

# Load argument
model = sys.argv[1]
id = sys.argv[2]

# Create the placeholder
images = np.zeros((10, 28, 28))

# Read and preprocess the image
img_gray = cv2.imread("./id/%s.png" % id, 0)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Extract features from the gray image
_, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3, 3), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

# Find contours
_, ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1])

for i, rect in enumerate(sorted_ctrs):
    # Find bounds
    x, y, w, h = cv2.boundingRect(rect)
    leng = int(h * 1.6)
    pt1 = int(y + h // 2 - leng // 2)
    pt2 = int(x + w // 2 - leng // 2)

    # Adjust image
    roi = img_dilation[pt1:pt1 + leng, pt2:pt2 + leng]
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # Save image
    images[i] = roi

# Reshape for model input
images = np.array(images).reshape(10, 28, 28, 1)

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
saver = tf.train.Saver()
saver.restore(session, "./models/%s_nn/%s_nn" % (model, model))

# Classify unseen digits
classifications = session.run(tf.argmax(Y, 1), feed_dict={X: images})

# Plot the digits and their classifications
plt.rc("image", cmap="gray")
fig = plt.figure(0)
fig.canvas.set_window_title("Student ID Identification")
for i in range(10):
    subplot = plt.subplot(1, 10, i+1)
    subplot.imshow(images[i].reshape(28, 28))
    subplot.text(0.5, -1.25, classifications[i], backgroundcolor=(0.0, 0.0, 0.0), color=(1.0, 1.0, 1.0), fontsize=22, ha="center", transform=subplot.transAxes)
    plt.xticks(())
    plt.yticks(())
plt.tight_layout()
plt.show()
