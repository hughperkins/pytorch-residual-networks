--[[
Copyright (c) 2016 Michael Wilber, Hugh Perkins 2016

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
  claim that you wrote the original software. If you use this software
  in a product, an acknowledgement in the product documentation would be
  appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
  misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

History:
- originally written by Michael Wilber, to run directly from lua/torch
- modified by Hugh Perkins, to run from python, via pytorch
   (basically, ripped out all the data loading, preprocessing, and a bunch of the logic around setting
    learning rate etc; moved it into python side)
--]]

require 'os'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'
require 'residual_model'

local ResidualTrainer = torch.class('ResidualTrainer')

function ResidualTrainer.__init(self, num_layer_groups)
  self.num_layer_groups = num_layer_groups
  self.model = residual_model.create(num_layer_groups)
  self.model:cuda()

  self.loss = nn.ClassNLLCriterion()
  self.loss:cuda()

  self.sgdState = {
     learningRate  = "will be set later",
     weightDecay   = 1e-4,
     momentum    = 0.9,
     dampening   = 0,
     nesterov    = true
  }
  self.weights, self.gradients = self.model:getParameters()
end

function ResidualTrainer.loadFrom(self, filepath)
    print("Loading model from ".. filepath)
    cutorch.setDevice(1)
    self.model = torch.load(filepath)
    print "Done"

    local sgdStatePath = string.gsub(filepath, "model", "sgdState")
    print("Trying to load sgdState from "..sgdStatePath)
    collectgarbage(); collectgarbage(); collectgarbage()
    self.sgdState = torch.load(sgdStatePath)
    collectgarbage(); collectgarbage(); collectgarbage()
    print('loaded sgdState')
--    print("Got", self.sgdState.nSampledImages,"images")
end

function ResidualTrainer.trainBatch(self, learningRate, batchInputs, batchLabels)
   self.sgdState.learningRate = learningRate

  -- copy data to gpu
  local inputscu = batchInputs:cuda()
  local labelscu = batchLabels:cuda()

  collectgarbage(); collectgarbage();
  self.model:training()
  self.gradients:zero()
  local y = self.model:forward(inputscu)
  local loss_val = self.loss:forward(y, labelscu)
  local df_dw = self.loss:backward(y, labelscu)
  self.model:backward(inputscu, df_dw)

  optim.sgd(function()
               return loss_val, self.gradients
            end,
            self.weights,
            self.sgdState)
  return loss_val
end

function ResidualTrainer.predict(self, batchInputs)
  self.model:evaluate()
  local batchSize = batchInputs:size(1)
  collectgarbage(); collectgarbage();
  local y = self.model:forward(batchInputs:cuda()):float()
  local _, indices = torch.sort(y, 2, true)
  -- indices has shape (batchSize, nClasses)
  local top1 = indices:select(2, 1):byte()
  local top5 = indices:narrow(2, 1,5):byte()
  return {top1=top1, top5=top5}  -- becomes a python dict, containing the tensors
end

