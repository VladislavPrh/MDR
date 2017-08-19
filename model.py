import numpy as np
import scipy
import tensorflow as tf


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(name, shape):
    return tf.get_variable(name,shape=shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())


class Model(object):
    """CNN architecture:
       INPUT -> CONV -> RELU -> CONV -> RELU ->
       POOL -> CONV -> POOL -> FC -> RELU -> 5X SOFTMAX
    """ 
        
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            # Create placeholders for feed data into graph
            self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            self.y = tf.placeholder(tf.int32, shape=[None, 6])
            self.keep_prob = tf.placeholder(tf.float32)

            # First convolutional layer
            # 16 filters - size(5x5x3)
            W_conv1 = weight_variable("W_c1", [5, 5, 3, 16])
            b_conv1 = bias_variable("B_c1", [16])
            h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
            h_norm1 = tf.nn.local_response_normalization(h_conv1)
            h_drop1 = tf.nn.dropout(h_norm1, self.keep_prob)

            # Second convolutional layer
            # 32 filters - size(5x5x16)
            W_conv2 = weight_variable("W_c2", [5, 5, 16, 32])
            b_conv2 = bias_variable("B_c2", [32])
            h_conv2 = tf.nn.relu(conv2d(h_drop1, W_conv2) + b_conv2)
            h_norm2 = tf.nn.local_response_normalization(h_conv2)
            h_pool2 = max_pool_2x2(h_norm2)
            h_drop2 = tf.nn.dropout(h_pool2, self.keep_prob)

            # Third convolutional layer
            # 64 filters - size(5x5x32)
            W_conv3 = weight_variable("W_c3", [5, 5, 32, 64])
            b_conv3 = bias_variable("B_c3", [64])
            h_conv3 = tf.nn.relu(conv2d(h_drop2, W_conv3) + b_conv3)
            h_norm3 = tf.nn.local_response_normalization(h_conv3)
            h_pool3 = max_pool_2x2(h_norm3)
                
            # Reshape tensor from POOL layer for connection with FC
            h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*64]) 
            h_drop3 = tf.nn.dropout(h_pool3_flat, self.keep_prob)

            # Fully connected layer
            W_fc1 = weight_variable("W_fc1", [8 * 8 * 64, 1024])
            b_fc1 = bias_variable("B_fc1", [1024])
            h_fc1 = tf.nn.relu(tf.matmul(h_drop3, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            # Create variables for 5 softmax classifiers
            W1 = tf.get_variable(shape=[1024, 11], name="W1",initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable(shape=[1024, 11], name="W2",initializer=tf.contrib.layers.xavier_initializer())
            W3 = tf.get_variable(shape=[1024, 11], name="W3",initializer=tf.contrib.layers.xavier_initializer())
            W4 = tf.get_variable(shape=[1024, 11], name="W4",initializer=tf.contrib.layers.xavier_initializer())                   
            W5 = tf.get_variable(shape=[1024, 11], name="W5",initializer=tf.contrib.layers.xavier_initializer())

            # Create biases for 5 softmax classifiers
            b1 = bias_variable("B1", [11])
            b2 = bias_variable("B2", [11])
            b3 = bias_variable("B3", [11])
            b4 = bias_variable("B4", [11])
            b5 = bias_variable("B5", [11])

            # Create logits
            self.logits_1 = tf.matmul(h_fc1_drop, W1) + b1
            self.logits_2 = tf.matmul(h_fc1_drop, W2) + b2
            self.logits_3 = tf.matmul(h_fc1_drop, W3) + b3
            self.logits_4 = tf.matmul(h_fc1_drop, W4) + b4
            self.logits_5 = tf.matmul(h_fc1_drop, W5) + b5

            # Define L2 Regularization, lambda == 0.001
            regularizer = (0.001*tf.nn.l2_loss(W_conv1) + 0.001*tf.nn.l2_loss(W_conv2) + \
                             0.001*tf.nn.l2_loss(W_conv3) + 0.001*tf.nn.l2_loss(W_fc1) + \
                             0.001*tf.nn.l2_loss(W1) + 0.001*tf.nn.l2_loss(W2) + \
                             0.001*tf.nn.l2_loss(W3) + 0.001*tf.nn.l2_loss(W4) + \
                             0.001*tf.nn.l2_loss(W5))
                
            # Define cross entropy loss function 
            self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_1, labels=self.y[:, 1])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_2, labels=self.y[:, 2])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_3, labels=self.y[:, 3])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_4, labels=self.y[:, 4])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_5, labels=self.y[:, 5])) + regularizer)

            # Define optimizer. 
            # Starting learning rate == 0.05, decay_steps == 10000, decay_rate == 0.96
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.96)
            self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(
                                                    self.loss, global_step=global_step)
            # Create saver
            self.saver = tf.train.Saver()
