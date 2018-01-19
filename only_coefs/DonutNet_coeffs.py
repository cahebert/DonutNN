import tensorflow as tf
from util_DonutNN import *
import pandas as pd
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='train a NN to predict '
                                     'structure/chemical mapping data as a function of seq')
    parser.add_argument('-f', '--filename', help='name of input file')
    parser.add_argument('-resdir', '--resultdir', help='location of results')
    return parser.parse_args()


args = parse_args()

restore = None #'results/simulatedData_6layer_1.00e-04_hnqyzj_testerr428.479568-10000'
learning_rate = 1e-4
filename = args.filename
iters = 10000
Ntest = 100
batch_size = 50
save = True

results_dir = args.resultdir

# set kernel parameters (size)
w_shapes = [ [6, 6, 1, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 96], \
             [6, 6, 96, 48], \
             [3, 3, 48, 1] ]

b_shapes = [ [96], \
             [96], \
             [96], \
             [96], \
             [96], \
             [96], \
             [96], \
             [96], \
             [96], \
             [96], \
             [96], \
             [48], \
             [1] ]
#    w_shapes = [ [3, 3, 1, 96], \
#             [12, 12, 96, 192], \
#             [12, 12, 192, 192], \
#             [12, 12, 192, 96], \
#             [12, 12, 96, 48], \
#             [3, 3, 48, 1] ]
#
#    b_shapes = [ [96], \
#             [192], \
#             [192], \
#             [96], \
#             [48], \
#             [1] ]

def main():
      print 'reading data...'
      data = pd.read_pickle(filename)
      #data = data.head(500)

      print 'building cnn... {} layers, learning rate = {}, {} iters'.format(len(w_shapes),learning_rate,iters)
      
      name = '%s_conv_%dlayer_%.2e' % (os.path.splitext(os.path.basename(filename))[0],len(b_shapes), learning_rate)

      model = CNN(w_shapes, b_shapes, name, learning_rate)

      if restore is not None:
            print 'restoring model parameters from file...'
            model.restore(restore)
      else:
          print 'training model...'
          loss = model.train(data, iters, batch_size)
          np.savetxt('%s/%s.loss' % (results_dir, model.name), loss.T, delimiter='\t', header='train\ttest', comments='')

      print 'getting final predictions...'

      testdata = data.sample(Ntest,axis=0)
      indices = testdata.index.values
      testDonut = np.vstack([np.expand_dims(x,0) for x in testdata['psf'].values])
      testDonut = np.expand_dims(testDonut,-1) 
        
      testWfPred = np.reshape(model.test(testDonut).flatten(),[-1,64,64])

      dfList = []    
      errList = [] 
      for i in range(len(indices)):
          df = pd.DataFrame(columns = ['wavefront_pred'])
          dfList.append( df.append({'wavefront_pred':testWfPred[i]},ignore_index=True) )
          diff = testWfPred[i]-testdata['wavefront'].values[i]
          errList.append(np.sum(np.square(diff)))
      predDf = pd.concat(dfList, ignore_index=True)
      testdata['wavefront_pred'] = predDf.values

      test_err = np.sum(errList)/Ntest
      
      print 'final training error: %.4f' % test_err

      testdata.to_pickle(results_dir + '/' + model.name + '_with_predictions.p')

      if save:
            print 'saving model parameters to file %s...' % model.name
            model.save('_testerr%f' % test_err)

if __name__ == '__main__':
    main()
