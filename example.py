#  The idea of tensorflow v1 is the following:
#  When each variable or placeholder is created and each operator is called on the variable or placeholder,
#  it is just the computing graph is created. The actual operation is not done yet. That's why the placeholder is named
#  placeholder because it is just a place to let the input get in later.
#  
#  It is when the session run is called and we tell it which node is needed to be run, the dependent computing graph will be 
#  trigger and run. We get the output we want by peeking the nodes we want from the computing graph will build.
#  To train the variable (or weights in the sense of deep learning), we add a loss that we want to minimize and an optimizer to 
#  the computing graph. Again, nothing will happen until we trigger the session run using the optimizer node this time. In this
#  way the variables(or weights) will start to change in the way that loss is minimized iteratively. We save the variables(weights) by saving 
#  the whole computing graph and we will later load it in the future in order to use these variables(weights) and the computing graph again.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Load data set
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#Size of each batch
batch_size = 100
#Calculate how many batches there are
n_batch = mnist.train.num_examples // batch_size

#Define two placeholders
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#Create a simple neural network
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#Quadratic cost function
loss = tf.reduce_mean(tf.square(y-prediction))
#Use gradient descent
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#Results are stored in a boolean list
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))#argmax returns the position of the largest value in a one-dimensional tensor
# Seeking accuracy rate
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#castConvert Boolean type to floating point type, True is 1.0, False is 0

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21): #Training 21 cycles
        for batch in range(n_batch):  #Train all pictures at once
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #Get batch_size pictures
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        
        print("epoch： " + str(epoch) + ",Training Accuracy： " + str(train_acc) + ",Testing Accuracy： " + str(test_acc))
