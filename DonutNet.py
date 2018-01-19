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
    parser.add_argument('-arch', '--architecture', type=str, help='architecture of the model')
    parser.add_argument('-r', '--restore', default=None, help='restore model parameters')
    parser.add_argument('-lr', '--learningrate', type=float,default=1e-4, help='learning rate')
    parser.add_argument('-i', '--iters', default=10000, type=int, help='iterations')
    parser.add_argument('-s','--save', default=0, type=int,help='save parameters')
    parser.add_argument('-a','--activation',default=None,type=str)
    parser.add_argument('-m','--mask', default=0, type=int)
    parser.add_argument('-in','--inputs', default=2, type=int)
    return parser.parse_args()

def parse_architecture_string(layers):
    s = layers.strip('[]').replace(' ','')
    s = s.split('),(')
    layers = []
    for tup in s:
        l = tup.strip('()')
        a,b = l.split(']')
        _,b,c = b.split(',')
        shape = [int(i) for i in a.strip('[]').split(',')]
        stride = int(b)
        ltype = str(c.strip('\''))
        layers.append((shape,stride,ltype))
    return layers

args = parse_args()
#CNN architecture
layers = parse_architecture_string(args.architecture)
#restore parameters?
restore = args.restore
#learning rate
learning_rate = args.learningrate
#data filename
filename = args.filename
#training iterations
iters = args.iters
#how many examples to test at the end
Ntest = 100
#batch size for each training set
batch_size = 50
#using one or two input images per example
inputs=args.inputs
#save parameters and outputs?
save = bool(args.save)
#what activation function to use for the last layer (recommend none)
activation = args.activation
#where to save results
results_dir = args.resultdir
#use a mask (annulus) when calculating L2 distance btw images
mask = bool(args.mask)



def main():
      print 'reading data...'
      data = pd.read_pickle(filename)
      #data = data.head(500)

      if 'd' in [z for x,y,z in layers]:
          print 'building cnn with deconvolution... {} layers, learning rate = {}, {} iters'.format(len(layers),learning_rate,iters)

          name = '%s_deconv_%dlayer_%inputs_%lr_%iters' % (os.path.splitext(os.path.basename(filename))[0],len(layers), inputs, learning_rate, iters)
      else:
          print 'building all convolutional cnn... {} layers, learning rate = {}, {} iters'.format(len(layers),learning_rate,iters)

          name = '%s_conv_%dlayer_%inputs_%lr_%iters' % (os.path.splitext(os.path.basename(filename))[0],len(layers), inputs, learning_rate, iters)

      model = CNN(layers, name, learning_rate, batch_size, mask, inputs)

      if restore is not None:
            print 'restoring model parameters from file...'
            model.restore(restore)
      else:
          print 'training model...'
          loss = model.train(data, iters, batch_size,inputs)
          np.savetxt('%s/%s.loss' % (results_dir, model.name), loss.T, delimiter='\t', header='train\ttest', comments='')

      print 'getting final predictions...'

      testdata = data.sample(Ntest,axis=0)
      indices = testdata.index.values
     
      if inputs == 2: 
          all_donuts_x = np.vstack([np.expand_dims(x, 0) for x in testdata['psf'].values])
          all_donuts_i = np.vstack([np.expand_dims(x, 0) for x in testdata['intra_psf'].values])
          testDonut = np.stack((all_donuts_x,all_donuts_i),axis=-1)
      else:
          donuts = np.vstack([np.expand_dims(x, 0) for x in testdata['psf'].values])          
          testDonut = np.expand_dims(donuts,-1) 
    
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
