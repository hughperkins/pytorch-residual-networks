"""
Trains cifar on residual network

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
import random
import time
import readline
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
devMode = False
if 'DEVMODE' in os.environ and os.environ['DEVMODE'] == '1':
  devMode = True
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

def loadData(data_dir, num_datafiles):
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

  # load test data
  d = loadPickle(join(data_dir, 'test_batch'))
  NTest = d['data'].shape[0]
  testData = np.zeros((NTest, inputPlanes, inputWidth, inputHeight), np.float32)
  testLabels = np.zeros(NTest, np.uint8)
  data = d['data'].reshape(dataLength, inputPlanes, inputWidth, inputHeight)
  testData[:] = data
  testLabels[:] = d['labels']

  return NTrain, trainData, trainLabels, NTest, testData, testLabels


# load the lua class
ResidualTrainer = PyTorchHelpers.load_lua_class('residual_trainer.lua', 'ResidualTrainer')
residualTrainer = ResidualTrainer(num_layer_groups)
if loadFrom is not None:
  residualTrainer.loadFrom(loadFrom)
print('residualTrainer', residualTrainer)

NTrain, trainData, trainLabels, NTest, testData, testLabels = loadData(data_dir, num_datafiles)

print('data loaded :-)')

# I think the mean and std are over all data, altogether, not specific to planes or pixel location?
mean = trainData.mean()
std = trainData.std()

trainData -= mean
trainData /= std

testData -= mean
testData /= std

print('data normalized check new mean/std:')
print('  trainmean=%s trainstd=%s testmean=%s teststd=%s' %
      (trainData.mean(), trainData.std(), testData.mean(), testData.std()))

# now we just have to call the lua class I think :-)

batchesPerEpoch = NTrain // batchSize
if devMode:
  batchesPerEpoch = 3  # impatient developer :-P
epoch = 0
while True:
#  print('epoch', epoch)
  learningRate = epochToLearningRate(epoch)
  epochLoss = 0
#  batchInputs 
  last = time.time()
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

    batchInputs = np.zeros((batchSize, inputPlanes, inputWidth, inputHeight), dtype=np.float32)
    batchLabels = trainLabels[indexes]

    # translate (translate directly into batch images)
    for i in range(batchSize):
       srcIdx = indexes[i]
       xoffs, yoffs = random.randint(-4,4), random.randint(-4,4)
       batch_y = [max(1,   1 + yoffs), min(32, 32 + yoffs)]
       src_y = [max(1,   1 - yoffs), min(32, 32 - yoffs)]
       batch_x = [max(1,   1 + xoffs), min(32, 32 + xoffs)]
       src_x = [max(1,   1 - xoffs), min(32, 32 - xoffs)]
       xmin, xmax = max(1, xoffs),  min(32, 32+xoffs)

       batchInputs[i][:, batch_y[0]:batch_y[1], batch_x[0]:batch_x[1]] = \
         trainData[srcIdx][:, src_y[0]:src_y[1], src_x[0]:src_x[1]]

    # flip
    for i in range(batchSize):
      if random.randint(0,1) == 1:
        batchInputs[i] = np.fliplr(batchInputs[i].transpose(1,2,0)).transpose(2,0,1)

    if devMode:
      now = time.time()
      duration = now - last
      print('preprocess time', duration)
      last = now

    loss = residualTrainer.trainBatch(learningRate, batchInputs, batchLabels)
    print('  epoch %s batch %s/%s loss %s' %(epoch, b, batchesPerEpoch, loss))
    epochLoss += loss

    if devMode:
      now = time.time()
      duration = now - last
      print('batch time', duration)
      last = now

  # evaluate on test data
  numTestBatches = NTest // batchSize
  if devMode:
    numTestBatches = 3  # impatient developer :-P
  testNumTop1Right = 0
  testNumTop5Right = 0
  testNumTotal = numTestBatches * batchSize
  for b in range(numTestBatches):
    batchInputs = testData[b * batchSize:(b+1) * batchSize]
    batchLabels = testLabels[b * batchSize:(b+1) * batchSize]
    res = residualTrainer.predict(batchInputs)
    top1 = res['top1'].asNumpyTensor()
    top5 = res['top5'].asNumpyTensor()
    labelsTiled5 = np.tile(batchLabels.reshape(batchSize, 1), (1, 5))
    top1_correct = (top1 == batchLabels).sum()
    top5_correct = (top5 == labelsTiled5).sum()
    testNumTop1Right += top1_correct
    testNumTop5Right += top5_correct
#    print('correct top1=%s top5=%s', top1_correct, top5_correct)

  testtop1acc = testNumTop1Right / testNumTotal * 100
  testtop5acc = testNumTop5Right / testNumTotal * 100
  print('epoch %s trainloss=%s top1acc=%s top5acc=%s' %
        (epoch, epochLoss, testtop1acc, testtop5acc))
  epoch += 1

