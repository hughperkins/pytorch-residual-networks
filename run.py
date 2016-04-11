"""
This is being tested only on python 2.7 for now.  python 3.4 will surely follow in the future (or if
you create an appropriate pull request :-) )
"""
from __future__ import print_function, division
import platform
import sys
import os
from os import path
from os.path import join
import numpy as np
import PyTorchHelpers


opt = {}
opt['batchSize'] = 128
opt['iterSize'] = 1
opt['Nsize'] = 3
#opt['dataRoot'] = '/mnt/cifar'
opt['loadFrom'] = ''

#--    --batchSize     (default 128)    Sub-batch size
#--    --iterSize      (default 1)     How many sub-batches in each batch
#--    --Nsize        (default 3)     Model has 6*n+2 layers.
#--    --dataRoot      (default /mnt/cifar) Data root folder
#--    --loadFrom      (default "")    Model to load
#--    --experimentName  (default "snapshots/cifar-residual-experiment1")

data_dir = 'cifar-10-batches-py'
num_datafiles = 5
num_datafiles = 1 # cos I lack patience during dev :-P

inputPlanes = 3
inputWidth = 32
inputHeight = 32

pyversion = platform.python_version_tuple()[0]

def loadPickle(path):
  with open(path, 'rb') as f:
    if pyversion == 2:
      import cPickle
      return cPickle.load(f)
    else:
      import pickle
      # not tested, maybe works ok? (no, it doesnt:
      # "UnicodeDecodeError: 'ascii' codec can't decode byte 0x8b in position 6: ordinal not in range(128)")
      return pickle.load(f)

# load the lua class
ResidualTrainer = PyTorchHelpers.load_lua_class('residual_trainer.lua', 'ResidualTrainer')
residualTrainer = ResidualTrainer(opt)
print('residualTrainer', residualTrainer)

#trainDatas = []
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



# set learning rate (was in lua)
#   -- From https://github.com/bgshih/cifar.torch/blob/master/train.lua#L119-L128
#   local sgdState = self.sgdState
#   if sgdState.epochCounter < 80 then
#      sgdState.learningRate = 0.1
#   elseif sgdState.epochCounter < 120 then
#      sgdState.learningRate = 0.01
#   else
#      sgdState.learningRate = 0.001
#   end

