import tensorflow as tf
import random
import numpy as np
import string

results_dir = './results'

def weight_variable(shape,name='weights'):
    # outputs a weight variable of [shape] intialized with small Gaussian distributed numbers
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name='biases'):
    # outputs a bias variable of [shape] initialized to 0.1
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

def conv2d(x, W, strides=[1, 1, 1, 1]):
    # returns a convolution of input x with weight matrix W (?) using stride (default 1)
    padded_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    return tf.nn.conv2d(padded_x, W, strides=strides, padding='VALID', name='convolution')

def add_conv_layer(w_shape, b_shape, x_input,name,strides=[1,1,1,1]):
    # returns a ReLUd convolution layer using inputs to define the weights
    weight = weight_variable(w_shape)
    bias = bias_variable(b_shape)
    return weight, conv2d(x_input, weight, strides) + bias

def batch_norm(x,depth):
    ## complicated because want mean and var to not be trained on
    mean = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
    variance = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)

    b_mean, b_variance = tf.nn.moments(x, axes=[0,1,2]) #computes mean and variance of x

    mean.assign(b_mean)
    variance.assign(b_variance)

    ##offset
    beta = tf.Variable(tf.constant(0.0, shape=[depth]))
    ##scale
    gamma = tf.Variable(tf.constant(1.0, shape=[depth]))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon=1e-4, name='batchnorm')

def set_up_n_layers(w_shapes, b_shapes, x_input, n):
    ## returns weights of final layer
    weights = []
    weight, conv = add_conv_layer(w_shapes[0], b_shapes[0], x_input, name='conv1')
    bn_conv = tf.nn.relu ( batch_norm(conv, b_shapes[0][0]) )
    weights.append(weight)
    for i in range(1,n):
        weight, conv = add_conv_layer(w_shapes[i], b_shapes[i], bn_conv, name='conv{}'.format(i+1))
        bn_conv = tf.nn.relu( batch_norm(conv, b_shapes[i][0]) )
        weights.append(weight)
    return weights, bn_conv

class CNN():
    def __init__(self, w_shapes, b_shapes, name='', learning_rate=1e-5, deconv=False,batch_size=50):
        self.name = name + '_' + ''.join(random.choice(string.ascii_lowercase) for _ in range(6))

        self.sess = tf.Session()
        self.build(w_shapes, b_shapes, learning_rate, deconv, batch_size)
        self.sess.run(tf.global_variables_initializer())
        self.init_writer()
        self.i = 0

    def init_writer(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('log_simple_stats', self.sess.graph)

    def build(self, w_shapes, b_shapes, learning_rate, deconv,batch_size):

        self.donut = tf.placeholder(tf.float32,
                                  shape=[None, 100, 100, 1],name='donut')
        self.coef_truth = tf.placeholder(tf.float32, shape=[None, 13, 1], name='coefs')

        if not deconv:
            weights, result = set_up_n_layers(w_shapes, b_shapes, self.donut, n=len(b_shapes))
        else:
            weights, result = set_up_n_layers_with_deconv(w_shapes, b_shapes, self.donut, n=len(b_shapes))
        
        self.weight = weights
        self.coef_pred = tf.reshape(result,[-1,13,1])
        
        self.loss = tf.nn.l2_loss( self.coef_pred - self.coef_truth )
        self.train_summ = tf.summary.scalar("L2 loss", self.loss)

        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        #self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)


    def train(self, data, iters=10000, batch_size=100, test=False):
        all_donuts = data['psf']
        all_coef = data['aberrations']

        all_donuts = np.vstack([np.expand_dims(x, 0) for x in all_donuts.values])
        all_donuts = np.expand_dims(all_donuts,-1) 
        all_coef = np.vstack([np.expand_dims(x, 0) for x in all_coef.values])
        all_coef = np.expand_dims(all_wf,-1) 

        period = 50
        loss = np.zeros((1,int(np.ceil(iters/float(period)))))

        for i in range(iters):
            batch_idx = np.random.choice(np.shape(all_donuts)[0], batch_size)
            if i % period == 0:
                loss[0,i/period], _ = self.get_loss(all_donuts,all_coef, i)
                print 'step %d: loss %g' % (i, loss[0, i/period])
            
            self.sess.run([self.train_step], feed_dict={self.donut: all_donuts[batch_idx], \
                                                                self.coef_truth: all_coef[batch_idx]})
            self.i += 1
        return loss

    def get_loss(self, donut, wf, i_t, max_chunk=25):
        fetches = [self.loss, self.train_summ]
        if donut.shape[0] < max_chunk:
            result = self.sess.run(fetches, feed_dict={self.donut: donut, self.wf_truth: wf})
            self.writer.add_summary(result, i)
            loss = result[0]
            result = result[1]
        else:
            chunk_losses = []
            nchunks = int(np.ceil(float(donut.shape[0])/max_chunk))
            for i in range(nchunks):
                j, k = i * max_chunk, (i+1) * max_chunk
                loss,result = self.sess.run(fetches, feed_dict={self.donut: donut[j:k], self.wf_truth: wf[j:k]})
                chunk_losses.append(loss)
            loss = np.sum(chunk_losses)
        self.writer.add_summary(result, i_t)
        #self.write_weights_hist(i=i) ##??

        return loss, result

    def test(self, donut,max_chunk=25):
        if donut.shape[0] < max_chunk:
            return self.wavefront_pred.eval(session=self.sess,feed_dict={self.donut: donut})
        else:
            chunk_wfs = np.zeros((donut.shape[0],64,64,1))
            nchunks = int(np.ceil(float(donut.shape[0])/max_chunk))
            for i in range(nchunks):
                j, k = i * max_chunk, (i+1) * max_chunk
                pred = self.wavefront_pred.eval(session=self.sess, feed_dict={self.donut: donut[j:k]})
                chunk_wfs[j:k,:,:,:] = pred #tf.reshape(pred,[-1,64,64])
            return chunk_wfs

    def restore(self, filename):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, filename)

    def save(self, suffix=''):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.sess, '%s/%s%s' % (results_dir, self.name, suffix), global_step=self.i)
        return

    def __del__(self):
        self.writer.close()

    
