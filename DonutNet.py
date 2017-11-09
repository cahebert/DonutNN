import tensorflow as tf
from DonutNet_util.py import *
import pandas as ps

## parameters to set
stepLength = tf.placeholder(tf.float32)

## load data ##
data = pd.read_pickle('/Users/clairealice/Documents/Research/Burchat/DonutNN/simulatedData.p')
ind = data.sample(1,axis=0).index.values

x = data['psf'][i]
wf_truth = data['wavefront'][i]

x_image = tf.reshape(x, [-1, 100, 100, 1])

# set kernel parameters (size)
w_shapes = [ [3, 3, 1, 96], \
            [12, 12, 96, 192], \
            [12, 12, 192, 192], \
            [12, 12, 192, 96], \
            [12, 12, 96, 48], \
            [3, 3, 48, 1] ]

b_shapes = [ [96], \
            [192], \
            [192], \
            [96], \
            [48], \
            [1] ]

wf_pred = set_up_n_layers(w_shapes, b_shapes, x_image, n=6)

#check that the size of prediction is right
if not tf.equal(tf.shape(wf_pred),tf.shape(wf_truth)):
    print "predicted and truth wavefronts have different size"

l2Loss = tf.nn.l2_loss(wf_pred - wf_truth)
tf.summary.scalar("L2 loss", l2Loss)

train_step = tf.train.GradientDescentOptimizer(stepLength).minimize(l2Loss)

sess = tf.Session()

# merge summaries, start summary writer and session  
summaries = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

# initialize variables
sess.run(tf.global_variables_initializer())


