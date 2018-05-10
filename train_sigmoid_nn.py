import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load the dataset
mnist_data = input_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# Create the variables for the neural network
# X  =  28x28 input image
# Y_ =  10 possible outputs [0-9]
# W1 =  weights linking each node between input layer and hidden layer (initialized as random)
# B1 =  biases of each node in hidden layer (initialized as random)
# W2 =  weights linking each node between hidden layer and output layer (initialized as random)
# B2 =  biases of each node in output layer (initialized as random)
X  = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.random_normal([28*28, 30], stddev=0.1), name="W1")
B1 = tf.Variable(tf.random_normal([30], stddev=0.1), name="B1")
W2 = tf.Variable(tf.random_normal([30, 10], stddev=0.1), name="W2")
B2 = tf.Variable(tf.random_normal([10], stddev=0.1), name="B2")

# Create the neural network model from the created variables
# Also, create the training method used to optimize the network
# Using gradient descent to minize the cross entropy of the network
XX = tf.reshape(X, [-1, 28*28])
H = tf.nn.sigmoid(tf.add(tf.matmul(XX, W1), B1))
Ylogits = tf.add(tf.matmul(H, W2), B2)
Y = tf.nn.sigmoid(Ylogits)
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ylogits, labels=Y_)) * 100.0
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

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
for i in range(10000+1):
    batch_X, batch_Y = mnist_data.train.next_batch(100)
    
    if i % 100 == 0:
        a, e = session.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
        print("%d: accuracy: %.4f error: %.4f" % (i, a, e))
    if i % 500 == 0:
        a, e = session.run([accuracy, cross_entropy], feed_dict={X: mnist_data.test.images, Y_: mnist_data.test.labels})
        if a > max_accuracy: max_accuracy = a
        print("%d: ***** epoch %d ***** test accuracy: %.4f test error: %.4f" % (i, i*100//mnist_data.train.images.shape[0]+1, a, e))
    session.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

# Max accuracy obtained by the model
print("max test accuracy: %.4f" % max_accuracy)

# Save the trained model
path = "./models/sigmoid_nn"
saver = tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)
saver.save(session, "%s/sigmoid_nn" % path)
print("training completed!")
