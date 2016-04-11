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
--]]

require 'os'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'
require 'residual_model'

--opt = lapp[[
--    --batchSize     (default 128)    Sub-batch size
--    --iterSize      (default 1)     How many sub-batches in each batch
--    --Nsize        (default 3)     Model has 6*n+2 layers.
--    --dataRoot      (default /mnt/cifar) Data root folder
--    --loadFrom      (default "")    Model to load
--    --experimentName  (default "snapshots/cifar-residual-experiment1")
--]]
--print(opt)

local ResidualTrainer = torch.class('ResidualTrainer)

function ResidualTrainer.__init(self, opt) -- opt is a dictionary, with values as per above 'lapp' section
  self.opt = opt

  -- Residual network.
  -- Input: 3x32x32
  local N = opt.Nsize
  local model = nil
  if opt.loadFrom == nil or opt.loadFrom == "" then
    model = residual_model.create()
    model:cuda()
    --print(#model:forward(torch.randn(100, 3, 32,32):cuda()))
  else
    print("Loading model from "..opt.loadFrom)
    cutorch.setDevice(1)
    model = torch.load(opt.loadFrom)
    print "Done"
  end
  self.model = model

  loss = nn.ClassNLLCriterion()
  loss:cuda()
  self.loss = loss

  sgdState = {
    --- For SGD with momentum ---
    ----[[
    -- My semi-working settings
     learningRate  = "will be set later",
     weightDecay   = 1e-4,
    -- Settings from their paper
    --learningRate = 0.1,
    --weightDecay   = 1e-4,

     momentum    = 0.9,
     dampening   = 0,
     nesterov    = true,
     --]]
  }

  if opt.loadFrom ~= nil and opt.loadFrom ~= "" then
    print("Trying to load sgdState from "..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    sgdState = torch.load(""..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    print("Got", sgdState.nSampledImages,"images")
  end
  self.sgdState = sgdState

  local weights, gradients = model:getParameters()
  self.weights = weights
  self.gradients = gradients
end

function ResidualTrainer.forwardBackwardBatch(self, inputs, labels)
   local model = self.model
   local gradients = self.gradients
   local sgdState = self.sgdState
   local opt = self.opt

   model:training()
   gradients:zero()

   -- From https://github.com/bgshih/cifar.torch/blob/master/train.lua#L119-L128
   if sgdState.epochCounter < 80 then
      sgdState.learningRate = 0.1
   elseif sgdState.epochCounter < 120 then
      sgdState.learningRate = 0.01
   else
      sgdState.learningRate = 0.001
   end

   local loss_val = 0
   local N = opt.iterSize
   local inputs, labels
   for i=1,N do
--      inputs, labels = dataTrain:getBatch()
      inputs = inputs:cuda()
      labels = labels:cuda()
      collectgarbage(); collectgarbage();
      local y = model:forward(inputs)
      loss_val = loss_val + loss:forward(y, labels)
      local df_dw = loss:backward(y, labels)
      model:backward(inputs, df_dw)
   end
   loss_val = loss_val / N
   gradients:mul( 1.0 / N )

   return loss_val, gradients, inputs:size(1) * N
end

function ResidualTrainer.initTraining(self, epochSize)
  self.epochSize = epochSize

  local sgdState = self.sgdState

--  self:evalModel()
--  local d = Date{os.date()}
--  local modelTag = string.format("%04d%02d%02d-%d",
--    d:year(), d:month(), d:day(), torch.random())
  sgdState.epochSize = epochSize
  sgdState.epochCounter = sgdState.epochCounter or 0
  sgdState.nSampledImages = sgdState.nSampledImages or 0
  sgdState.nEvalCounter = sgdState.nEvalCounter or 0
  local whichOptimMethod = optim.sgd
  if sgdState.whichOptimMethod then
     whichOptimMethod = optim[sgdState.whichOptimMethod]
  end
end

--function ResidualTrainer.trainBatch(self)
--  local weights = self.weights
--  local sgdState = self.sgdState
--  local epochSize = self.epochSize
--  local gradients = self.gradients

--  collectgarbage(); collectgarbage()
--  -- Run forward and backward pass on inputs and labels
--  local loss_val, gradients, batchProcessed = self:forwardBackwardBatch()
--  -- SGD step: modifies weights in-place
--  whichOptimMethod(function() return loss_val, gradients end,
--                    weights,
--                    sgdState)
--  -- Display progress and loss
--  sgdState.nSampledImages = sgdState.nSampledImages + batchProcessed
--  sgdState.nEvalCounter = sgdState.nEvalCounter + 1
--  xlua.progress(sgdState.nSampledImages%epochSize, epochSize)

--  if math.floor(sgdState.nSampledImages / epochSize) ~= sgdState.epochCounter then
--    -- Epoch completed!
--    xlua.progress(epochSize, epochSize)
--    sgdState.epochCounter = math.floor(sgdState.nSampledImages / epochSize)
----    local testResults = self:evaluateModel(model, dataTest, opt.batchSize)
--    print("\n\n----- Epoch "..sgdState.epochCounter.." -----")
--  end
--end

function ResidualTrainer.predict(self, batch)
  local model = self.model

  print("Evaluating...")
  model:evaluate()
  local batchSize = labels:size(1)
  collectgarbage(); collectgarbage();
  local y = model:forward(batch:cuda()):float()
  local _, indices = torch.sort(y, 2, true)
  -- indices has shape (batchSize, nClasses)
  local top1 = indices:select(2, 1)
  local top5 = indices:narrow(2, 1,5)
  return top1, top5
end

