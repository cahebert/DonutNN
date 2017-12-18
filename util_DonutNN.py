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
    # returns a convolution of input x with weight matrix W using stride (default 1)
    padded_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    return tf.nn.conv2d(padded_x, W, strides=strides, padding='VALID', name='convolution')

def add_conv_layer(w_shape, b_shape, x_input,name,strides=[1,1,1,1]):
    # returns a convolution layer using inputs to define the weights
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
    ## returns weights and layers of NN
    weights = []
    layers = []
    weight, conv = add_conv_layer(w_shapes[0], b_shapes[0], x_input, name='conv1')
    bn_conv = tf.nn.crelu ( batch_norm(conv, b_shapes[0][0]) )
    weights.append(weight)
    layers.append(bn_conv)
    for i in range(1,n-1):
        weight, conv = add_conv_layer(w_shapes[i], b_shapes[i], bn_conv, name='conv{}'.format(i+1))
        bn_conv = tf.nn.crelu( batch_norm(conv, b_shapes[i][0]) )
        weights.append(weight)
        layers.append(bn_conv)
    weight, conv = add_conv_layer(w_shapes[-1], b_shapes[-1], bn_conv, name='conv{}'.format(n))
    bn_conv = batch_norm(conv,b_shapes[-1][0])
    weights.append(weight)
    layers.append(bn_conv)
    return weights, layers


def deconv2d(x, W, output_shape, strides=[1, 1, 1, 1]):
    ## returns deconv layer
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=strides, padding='VALID',name='deconv') 

def add_deconv_layer(w_shape, b_shape, output_shape, x_input, name, strides=[1,1,1,1]):
    # returns a deconvolution layer with output_shape using inputs to define the weights
    weight = weight_variable(w_shape)
    bias = bias_variable(b_shape)
    return weight, deconv2d(x_input, weight, output_shape, strides=strides) + bias

def set_up_n_layers_with_deconv(w_shapes, b_shapes, x_input, n):
    ## returns weights and layers
    weights = []
    layers = []
    strds = [[1,2,2,1],[1,1,1,1],[1,2,2,1]]

    weight, conv = add_conv_layer(w_shapes[0], b_shapes[0], x_input, name='conv1',strides=strds[0])
    bn_conv = tf.nn.crelu( batch_norm(conv, b_shapes[0][0]) )
    weights.append(weight)
    layers.append(bn_conv)

    for i in range(1,3):
        weight, conv = add_conv_layer(w_shapes[i], b_shapes[i], bn_conv, name='conv{}'.format(i),strides=strds[i])
        bn_conv = tf.nn.crelu( batch_norm(conv, b_shapes[i][0]) )
        weights.append(weight)
        layers.append(bn_conv)

    batch_size = tf.shape(x_input)[0]
    output_shapes = [[batch_size, 26, 26, 96], [batch_size, 31, 31, 96], [batch_size, 64, 64, 48]]
    for i in range(3,5):
        weight, deconv = add_deconv_layer(w_shapes[i], b_shapes[i], output_shapes[i-3], bn_conv, name='deconv{}'.format(i))
        bn_conv = tf.nn.crelu( batch_norm(deconv, b_shapes[i][0]) )
        weights.append(weight)
        layers.append(bn_conv)

    weight, deconv = add_deconv_layer(w_shapes[5], b_shapes[5], output_shapes[2], bn_conv, name='deconv5', strides=[1,2,2,1])
    bn_conv = tf.nn.crelu( batch_norm(deconv, b_shapes[5][0]) )
    weights.append(weight)    
    layers.append(bn_conv)

    weight, conv = add_conv_layer(w_shapes[6], b_shapes[6], bn_conv, name='conv6')
    weights.append(weight)
    bn_conv = batch_norm(conv,b_shapes[6][0])
    layers.append(bn_conv)

    return weights, layers

class CNN():
    def __init__(self, w_shapes, b_shapes, name='', learning_rate=1e-4, deconv=False, batch_size=50, mask=True):
        self.name = name + '_' + ''.join(random.choice(string.ascii_lowercase) for _ in range(6))

        self.sess = tf.Session()
        self.build(w_shapes, b_shapes, learning_rate, deconv, batch_size, mask)
        self.sess.run(tf.global_variables_initializer())
        self.init_writer()
        self.i = 0

    def init_writer(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('log_simple_stats', self.sess.graph)

    def build(self, w_shapes, b_shapes, learning_rate, deconv, batch_size, activation, mask):

        self.donut = tf.placeholder(tf.float32,
                                  shape=[None, 100, 100, 1],name='donut')
        self.wf_truth = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='wavefront')

        if not deconv:
            weights, layers = set_up_n_layers(w_shapes, b_shapes, self.donut, n=len(b_shapes))
        else:
            weights, layers = set_up_n_layers_with_deconv(w_shapes, b_shapes, self.donut, n=len(b_shapes))
        
        self.weight = weights
        self.layers = layers
        final_output = layers[-1]
        self.wavefront_pred = tf.reshape(final_output,[-1,64,64,1])

        if mask:
            nx, ny = 64,64
            x = np.linspace(-nx/2,nx/2,nx)
            y = np.linspace(-nx/2,nx/2,nx)
            X,Y = np.meshgrid(x,y)

            R = np.hypot(X,Y)
            annulus = ( (R<nx/2) & (R>nx*.3 ) )       
 
            mask = tf.convert_to_tensor(annulus, dtype=bool)
            mask_t = tf.stack([mask]*batch_size)

            diff = tf.boolean_mask(self.wavefront_pred,mask_t) - tf.boolean_mask(self.wf_truth,mask_t)
        else:
            diff = self.wavefront_pred-self.wf_truth

        self.loss = tf.nn.l2_loss( diff )
        self.train_summ = tf.summary.scalar("L2 loss", self.loss)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


    def train(self, data, iters=10000, batch_size=100, test=False):
        all_donuts = data['psf']
        all_wf = data['wavefront']

        all_donuts = np.vstack([np.expand_dims(x, 0) for x in all_donuts.values])
        all_donuts = np.expand_dims(all_donuts,-1) 
        all_wf = np.vstack([np.expand_dims(x, 0) for x in all_wf.values])
        all_wf = np.expand_dims(all_wf,-1) 

        period = 25
        loss = np.zeros((1,int(np.ceil(iters/float(period)))))

        for i in range(iters):
            batch_idx = np.random.choice(np.shape(all_donuts)[0], batch_size)
            
            if i % period == 0:
                loss[0,i/period], _ = self.get_loss(all_donuts,all_wf, i)
                print 'step %d: loss %g' % (i, loss[0, i/period])
                print 'min value of wf: ', np.min(self.wavefront_pred.eval(session=self.sess,feed_dict={self.donut: all_donuts[i:i+1]}))
            self.sess.run([self.train_step], feed_dict={self.donut: all_donuts[batch_idx], \
                                                                self.wf_truth: all_wf[batch_idx]})
            self.i += 1
        return loss

    def get_loss(self, donut, wf, i_t, max_chunk=50):
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

    def test(self, donut,max_chunk=50):
        if donut.shape[0] < max_chunk:
            return self.wavefront_pred.eval(session=self.sess,feed_dict={self.donut: donut})
        else:
            chunk_wfs = np.zeros((donut.shape[0],64,64,1))
            nchunks = int(np.ceil(float(donut.shape[0])/max_chunk))
            for i in range(nchunks):
                j, k = i * max_chunk, (i+1) * max_chunk
                pred = self.wavefront_pred.eval(session=self.sess, feed_dict={self.donut: donut[j:k]})
                chunk_wfs[j:k,:,:,:] = pred 
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

    
