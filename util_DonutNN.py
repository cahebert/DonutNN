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

def add_conv_layer(w_shape, x_input,name,strides=[1,1,1,1]):
    # returns a convolution layer using inputs to define the weights
    weight = weight_variable(w_shape)
    bias = bias_variable([w_shape[-1]])
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

def deconv2d(x, W, output_shape, strides=[1, 1, 1, 1]):
    ## returns deconv layer
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=strides, padding='VALID',name='deconv') 

def add_deconv_layer(w_shape, output_shape, x_input, name, strides=[1,1,1,1]):
    # returns a deconvolution layer with output_shape using inputs to define the weights
    weight = weight_variable(w_shape)
    bias = bias_variable([w_shape[2]])
    return weight, deconv2d(x_input, weight, output_shape, strides=strides) + bias

def setup_layers(x_input, layers):
    n = len(layers)
    n_img = 100
    weights = []
    layers_out = []

    for i in range(n):
        w, s, l = layers[i]
        if i == 0:
            weight, conv = add_conv_layer(w, x_input, name='conv1',strides=[1,s,s,1])
            bn_s = w[-1]
            n_img = (n_img - w[1] + 2.)/s + 1
            assert n_img%int(n_img)==0, 'parameters return non-int image shape'
        else:
            w[2] = 2*w[2]
            if l == 'c':
                weight, conv = add_conv_layer(w, bn_conv, name='conv{}'.format(i), strides=[1,s,s,1])
                bn_s = w[-1]
                n_img = (n_img - w[1] + 2.)/s + 1
                assert n_img%int(n_img)==0, 'parameters return non-int image shape'

            elif l == 'd':
                temp = w[-1]
                w[-1] = w[2]
                w[2] = temp

                n_img = s*(n_img-1) + w[1]
                assert n_img%int(n_img)==0, 'parameters return non-int image shape'
                batchsize = tf.shape(x_input)[0]
                output_shape = tf.pack([batchsize, int(n_img), int(n_img), w[2]])
                weight, conv = add_deconv_layer(w, output_shape, bn_conv, name='deconv{}'.format(i), strides=[1,s,s,1])
                bn_s = w[2]
        if i == n-1:
            bn_conv = batch_norm(conv, bn_s)
        else:
            bn_conv = tf.nn.crelu( batch_norm(conv, bn_s) )
        weights.append(weight)
        layers_out.append(bn_conv)
    return weights, layers_out

class CNN():
    def __init__(self, layers, name='', learning_rate=1e-4, batch_size=100, mask=True, inputs=2):
        self.name = name + '_' + ''.join(random.choice(string.ascii_lowercase) for _ in range(6))

        self.sess = tf.Session()
        self.build(layers, learning_rate, mask, inputs)
        self.sess.run(tf.global_variables_initializer())
        self.init_writer()
        self.i = 0

    def init_writer(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('log_simple_stats', self.sess.graph)

    def build(self, layers, learning_rate, mask, inputs):
        self.donut = tf.placeholder(tf.float32,
                                  shape=[None, 100, 100, inputs],name='donut')
        self.wf_truth = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='wavefront')

        weights, layers_out = setup_layers(self.donut, layers)

        self.weight = weights
        self.layers = layers_out
        final_output = layers_out[-1]
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


    def train(self, data, iters=10000, batch_size=100, test=False,inputs=2):
        if inputs==2:
            all_donuts_x = data['psf']
            all_donuts_i = data['intra_psf']
            all_wf = data['wavefront']

            all_donuts_x = np.vstack([np.expand_dims(x, 0) for x in all_donuts_x.values])
            all_donuts_i = np.vstack([np.expand_dims(x, 0) for x in all_donuts_i.values])
            all_donuts = np.stack((all_donuts_x,all_donuts_i),axis=-1)
            print np.shape(all_donuts)        
        else:
            all_donuts = data['psf']
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

    
