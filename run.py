"""
This is being tested only on python 2.7 for now.  python 3.4 will surely follow in the future (or if
you create an appropriate pull request :-) )

Usage:
  run.py [options]

Options:
  --batchsize BATCHSIZE      batchsize [default: 128]
  --loadfrom LOADFROM        load from this model file [default: None]
  --numlayergroups NUMLAYERGROUPS    number layer groups [default: 3]
"""

from __future__ import print_function, division
import platform
import sys
import os
from os import path
from os.path import join
from docopt import docopt
import numpy as np
import PyTorchHelpers
pyversion = int(platform.python_version_tuple()[0])
if pyversion == 2:
  import cPickle
else:
  import pickle


args = docopt(__doc__)
batchSize = int(args['--batchsize'])
loadFrom = args['--loadfrom']
if loadFrom == 'None':
  loadFrom = None
num_layer_groups = int(args['--numlayergroups'])

data_dir = 'cifar-10-batches-py'
num_datafiles = 5
num_datafiles = 1 # cos I lack patience during dev :-P

inputPlanes = 3
inputWidth = 32
inputHeight = 32

def loadPickle(path):
  with open(path, 'rb') as f:
    if pyversion == 2:
      return cPickle.load(f)
    else:
      return {k.decode('utf-8'): v for k,v in pickle.load(f, encoding='bytes').items()}

def epochToLearningRate(epoch):
   # From https://github.com/bgshih/cifar.torch/blob/master/train.lua#L119-L128
   if epoch < 80:
      return 0.1
   if epoch < 120:
      return 0.01
   return 0.001


# load the lua class
ResidualTrainer = PyTorchHelpers.load_lua_class('residual_trainer.lua', 'ResidualTrainer')
residualTrainer = ResidualTrainer(num_layer_groups)
if loadFrom is not None:
  residualTrainer.loadFrom(loadFrom)
print('residualTrainer', residualTrainer)

# load training data
trainData = None
trainLabels = None
NTrain = None
for i in range(num_datafiles):
  d = loadPickle(join(data_dir, 'data_batch_%s' % (i+1)))
  dataLength = d['data'].shape[0]
  NTrain = num_datafiles * dataLength
  if trainData is None:
    trainData = np.zeros((NTrain, inputPlanes, inputWidth, inputHeight), np.float32)
    trainLabels = np.zeros(NTrain, np.uint8)
  data = d['data'].reshape(dataLength, inputPlanes, inputWidth, inputHeight)
  trainData[i * dataLength:(i+1) * dataLength] = data
  trainLabels[i * dataLength:(i+1) * dataLength] = d['labels']

print('data loaded :-)')

# I think the mean and std are over all data, altogether, not specific to planes or pixel location?
mean = trainData.mean()
std = trainData.std()

print('mean', mean, 'std', std)

trainData -= mean
trainData /= std

# now we just have to call the lua class I think :-)

batchesPerEpoch = NTrain // batchSize
epoch = 0
while True:
  print('epoch', epoch)
  learningRate = epochToLearningRate(epoch)
  epochLoss = 0
  epochNumRight = 0
  for b in range(batchesPerEpoch):
    # we have to populate batchInputs and batchLabels :-(
    # seems there is a bunch of preprocessing to do :-P
    # https://github.com/gcr/torch-residual-networks/blob/bc1bafff731091bb382bece58d8252291bfbf206/data/cifar-dataset.lua#L56-L75

    # so we have to do:
    # - randomly sample batchSize inputs, with replacement (both between batches, and within batches)
    # - random translate by up to 4 horiz (+ve/-ve) and vert (+ve/-ve)  (in the paper, this is described as
    #   adding 4-padding, then cutting 32x32 patch)
    # - randomly flip horizontally

    # draw samples
    indexes = np.random.randint(NTrain, size=(batchSize))
    batchInputs = trainData[indexes]
    batchLabels = trainLabels[indexes]

    # TODO: translate
    xOffs = np.random.randint(-4, 4, size=(batchSize))
    yOffs = np.random.randint(-4, 4, size=(batchSize))

    # TODO: flip

    loss = residualTrainer.trainBatch(learningRate, batchInputs, batchLabels)
    print('  epoch %s batch %s/%s loss %s' %(epoch, b, batchesPerEpoch, loss))
    epochLoss += loss
  print('finished epoch', epoch, 'loss', epochLoss)
  epoch += 1

