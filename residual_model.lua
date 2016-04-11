require 'nn'
require 'cudnn'
require 'nngraph'
require 'residual_layers'
local nninit = require 'nninit'

residual_model = {}

local L = residual_model

function L.create(N)
  local input = nn.Identity()()
  ------> 3, 32,32
  local model = cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
      :init('weight', nninit.kaiming, {gain = 'relu'})
      :init('bias', nninit.constant, 0)(input)
  model = cudnn.SpatialBatchNormalization(16)(model)
  model = cudnn.ReLU(true)(model)
  ------> 16, 32,32  First Group
  for i=1,N do  model = residual_layers.addResidualLayer2(model, 16)  end
  ------> 32, 16,16  Second Group
  model = residual_layers.addResidualLayer2(model, 16, 32, 2)
  for i=1,N-1 do  model = residual_layers.addResidualLayer2(model, 32)  end
  ------> 64, 8,8  Third Group
  model = residual_layers.addResidualLayer2(model, 32, 64, 2)
  for i=1,N-1 do  model = residual_layers.addResidualLayer2(model, 64)  end
  ------> 10, 8,8  Pooling, Linear, Softmax
  model = nn.SpatialAveragePooling(8,8)(model)
  model = nn.Reshape(64)(model)
  model = nn.Linear(64, 10)(model)
  model = nn.LogSoftMax()(model)

  model = nn.gModule({input}, {model})
  return model
end

return L

