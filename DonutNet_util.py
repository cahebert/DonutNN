import tensorflow as tf

def weight_variable(shape):
    # outputs a weight variable of [shape] intialized with small Gaussian distributed numbers
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # outputs a bias variable of [shape] initialized to 0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, strides=[1, 1, 1, 1]):
    # returns a convolution of input x with weight matrix W (?) usins stride (default 1)
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def add_conv_ReLU_layer(w_shape, b_shape, x_input):
    # returns a ReLUd convolution layer using inputs to define the weights
    weight = weight_variable(w_shape)
    bias = bias_variable(b_shape)
    return tf.nn.relu(conv2d(x_input, weight) + bias)

def set_up_n_layers(weights, biases, x_input, n)
    # returns weights of final layer
    conv = add_conv_ReLU_layer(w_shapes[0], b_shapes[0], x_input)
    for i in range(1,n):
        conv = add_conv_ReLU_layer(w_shapes[i], b_shapes[i], conv)
    return conv


## try without these first? ##
def max_pool_kbyk(x, k=2, strides=[1, 2, 2, 1]):
    # returns a pooled x shrunk by k (default 2) on each side
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=strides, padding='SAME')

def deconv2d(x, W, output_shape, strides=[1, 1, 1, 1]):
    # upsamples x
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=strides, padding='SAME') 
    # how to choose output shape? # # read web page for output shape and weights details #
    