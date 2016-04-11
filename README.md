# Deep Residual Learning for Image Recognition

STATUS: DRAFT, IN PROGRESS. For the real (torch/lua) version, see https://github.com/gcr/torch-residual-networks/network for now please :-)

This is a pytorch implementation of ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385) the winners of the 2015 ILSVRC and COCO challenges.

It's forked from Michael Wilber's torch-residual-networks.  The data loading and preprocessing have been moved from
the lua side into the python side, so you're free to load different data, change the preprocessing, and so on.

For full readme please see https://github.com/gcr/torch-residual-networks/network (I've removed it from here, so it is clear who 'I' is :-) )

## How to use

- You need at least CUDA 7.0 and CuDNN v4.
- Install Torch.
- Install the Torch CUDNN V4 library: `git clone https://github.com/soumith/cudnn.torch; cd cudnn; git co R4; luarocks make` This will give you `cudnn.SpatialBatchNormalization`, which helps save quite a lot of memory.
- Install nninit: `luarocks install nninit`.
- Setup python (tested on 2.7 for now; 3.4 will follow):
```
virtualenv -p python27 env27
source env27/bin/activate
pip install numpy
```
- Install pytorch:
```
pushd ..
git clone https://github.com/hughperkins/pytorch
cd pytorch
./build.sh
popd
```
- Download cifar dataset, simply run: `./download.sh`
- Run `python run.py`

## Changes

2016 April 16:
- first forked from https://github.com/gcr/torch-residual-networks/network

