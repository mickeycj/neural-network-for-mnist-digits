import math
import os
import tensorflow as tf
from mnist import read_dataset

# Load the dataset
mnist_data = read_dataset()

# Create variable learning rate
step = tf.placeholder(tf.int32)

# Create the variables for the neural network
# X  =  28x28 input image
# Y_ =  10 possible outputs [0-9]
# W1 =  weights linking each node between the input layer and the first convolutional layer (initialized as truncated)
# B1 =  biases of each node in the first convolutional layer (initialized as ones)
# W2 =  weights linking each node between the first convolutional layer and the second convolutional layer (initialized as truncated)
# B2 =  biases of each node in the second convolutional layer (initialized as ones)
# W3 =  weights linking each node between the second convolutional layer and the third convolutional layer (initialized as truncated)
# B3 =  biases of each node in the third convolutional layer (initialized as ones)
# W4 =  weights linking each node between the third convolutional layer and the hidden layer (initialized as truncated)
# B4 =  biases of each node in the hidden layer (initialized as ones)
# W5 =  weights linking each node between the hidden layer and the output layer (initialized as truncated)
# B5 =  biases of each node in the output layer (initialized as ones)
X  = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1), name="W1")
B1 = tf.Variable(tf.ones([4])/10, name="B1")
W2 = tf.Variable(tf.truncated_normal([5, 5, 4, 8], stddev=0.1), name="W2")
B2 = tf.Variable(tf.ones([8])/10, name="B2")
W3 = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev=0.1), name="W3")
B3 = tf.Variable(tf.ones([12])/10, name="B3")
W4 = tf.Variable(tf.truncated_normal([7*7*12, 200], stddev=0.1), name="W4")
B4 = tf.Variable(tf.ones([200])/10, name="B4")
W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1), name="W5")
B5 = tf.Variable(tf.ones([10])/10, name="B5")

# Create the neural network model from the created variables
# Also, create the training method used to optimize the network
# using Adam algorithm to minize the cross entropy of the network
C1 = tf.nn.relu(tf.add(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME'), B1))
C2 = tf.nn.relu(tf.add(tf.nn.conv2d(C1, W2, strides=[1, 2, 2, 1], padding='SAME'), B2))
C3 = tf.nn.relu(tf.add(tf.nn.conv2d(C2, W3, strides=[1, 2, 2, 1], padding='SAME'), B3))
CC = tf.reshape(C3, [-1, 7*7*12])
H = tf.nn.relu(tf.add(tf.matmul(CC, W4), B4))
Ylogits = tf.add(tf.matmul(H, W5), B5)
Y = tf.nn.softmax(Ylogits)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_))*100.0
learning_rate = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Specify how the accuracy is calculated
correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize TensorFlow session
initializer = tf.global_variables_initializer()
session = tf.Session()
session.run(initializer)

# Perform training of the neural network
# 100 samples for each batch
# Evaluate the performance on testing set every 500 samples (5 batches)
max_accuracy = 0
for i in range(5000+1):
    batch_X, batch_Y = mnist_data.train.next_batch(100)
    
    if i % 100 == 0:
        a, e = session.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, step: i})
        print("%d: accuracy: %.4f error: %.4f" % (i, a, e))
    if i % 500 == 0:
        a, e = session.run([accuracy, cross_entropy], feed_dict={X: mnist_data.test.images, Y_: mnist_data.test.labels})
        if a > max_accuracy: max_accuracy = a
        print("%d: ***** epoch %d ***** test accuracy: %.4f test error: %.4f" % (i, i*100//mnist_data.train.images.shape[0]+1, a, e))
    session.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, step: i})

# Max accuracy obtained by the model
print("max test accuracy: %.4f" % max_accuracy)

# Save the trained model
path = "./models/convolutional_nn"
saver = tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)
saver.save(session, "%s/convolutional_nn" % path)
print("training completed!")
