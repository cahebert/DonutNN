import tensorflow as tf
from DonutNet_util import *
import pandas as pd
import numpy as np
import os


restore = None
learning_rate = 1e-5
filename = '/Users/clairealice/Documents/Research/Burchat/DonutNN/simulatedData.p'
iters = 100
Ntest = 10
batch_size = 25
save = True

results_dir = '/Documents/Research/Burchat/DonutNN'

# set kernel parameters (size)
# w_shapes = [ [3, 3, 1, 96], \
#             [12, 12, 96, 192], \
#             [12, 12, 192, 192], \
#             [12, 12, 192, 96], \
#             [12, 12, 96, 48], \
#             [3, 3, 48, 1] ]

# b_shapes = [ [96], \
#             [192], \
#             [192], \
#             [96], \
#             [48], \
#             [1] ]

w_shapes = [ [21, 21, 1, 4],\
            [21, 21, 4, 4],\
            [3, 3, 4, 1] ]

b_shapes = [ [4], [4], [1] ]


def main():
      print 'reading data...'
      data = pd.read_pickle(filename)
      data = data.head(500)

      print 'building cnn...'
      name = '%s_%dlayer_%.2e' % (os.path.splitext(os.path.basename(filename))[0],len(b_shapes), learning_rate)
      
      model = CNN(w_shapes, b_shapes, name, learning_rate)

      if restore is not None:
            print 'restoring model parameters from file...'
            model.restore(restore)

      print 'training model...'
      loss = model.train(data, iters, batch_size)

      np.savetxt('%s/%s.loss' % (results_dir, model.name), loss.T, delimiter='\t', header='train\ttest', comments='')

      print 'getting final predictions...'


      testdata = data.sample(Ntest,axis=0)
      testDonut = testdata['psf']
      testDonut = np.vstack([np.expand_dims(x,0) for x in testDonut.values])
      testDonut = np.expand_dims(testDonut,-1) 
        
      testdata['wavefront_predicted'] = model.test(testDonut)

      err = testdata['wavefront'] - testdata['wavefront_predicted']
      train_err = np.sum(np.square(err))/Ntest
      print 'final training error: %.4f' % train_err

      testdata.to_pickle(results_dir + '/' + model.name + '_with_predictions.p')

      ###lots more to add here###
      #should save model parameters
      if save:
            print 'saving model parameters to file %s...' % model.name
            model.save('_trainerr%f' % train_err)
      

if __name__ == '__main__':
    main()