import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load the dataset
mnist_data = input_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# Create each layer of the neural network
# X  =  28x28 input image
# Y_ =  10 possible outputs [0-9]
# W  =  weights linking each node between layers (initialized as zeros)
# B  =  biases of each node (initialized as zeros)
# XX =  input layer with 784 (28x28) nodes
X  = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
W  = tf.Variable(tf.zeros([28*28, 10]))
B  = tf.Variable(tf.zeros([10]))
XX = tf.reshape(X, [-1, 28*28])

# Create the neural network model from the create layers
# Also, create the training method used to optimize the network
# Using gradient descent to minize the square error of the network
Y = tf.nn.sigmoid(tf.add(tf.matmul(XX, W), B))
square_error = tf.reduce_mean(tf.squared_difference(Y_, Y)) * 1000.0
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(square_error)

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
        a, e = session.run([accuracy, square_error], feed_dict={X: batch_X, Y_: batch_Y})
        print("%d: accuracy: %.4f error: %.4f" % (i, a, e))
    if i % 500 == 0:
        a, e = session.run([accuracy, square_error], feed_dict={X: mnist_data.test.images, Y_: mnist_data.test.labels})
        if a > max_accuracy: max_accuracy = a
        print("%d: ***** epoch %d ***** test accuracy: %.4f test error: %.4f" % (i, i*100//mnist_data.train.images.shape[0]+1, a, e))
    session.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

# Max accuracy obtained by the model
print("max test accuracy: %.4f" % max_accuracy)

# Save the trained model
print("training completed!")
saver = tf.train.Saver()
saver.save(session, "./models/neural_network/sigmoid_nn")
