# Deep Residual Learning for Image Recognition

This is a pytorch implementation of ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385) the winners of the 2015 ILSVRC and COCO challenges.

It's forked from Michael Wilber's [torch-residual-networks](https://github.com/gcr/torch-residual-networks) .
The data loading and preprocessing have been moved from
the lua side into the python side, so you can easily modify the data loading and preprocessing, using the python
tools and libraries you're used to using.

For full readme on the original torch-residual-networks library,
please see https://github.com/gcr/torch-residual-networks/network

## How to use

- You need at least CUDA 7.0 and CuDNN v4
- Install Torch:
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch
# install dependencies.  To install everything:
  bash install-deps
# Or, if you're on ubuntu, you only need the following dependencies:
  sudo apt-get update -y
  sudo apt-get install -y wget git gcc g++ cmake libffi-dev \
       libblas-dev liblapack-dev libatlas-base-dev gfortran libreadline-dev
# install torch
./install.sh
```
- install torch cudnn and nninit:
```
luarocks install cudnn
luarocks install nninit
```
- Setup python (tested on 2.7 for now; 3.4 will follow):
```
sudo apt-get install python2.7-dev
virtualenv -p python2.7 ~/env27
source ~/env27/bin/activate
pip install docopt
pip install numpy
```
- Install pytorch:
```
git clone https://github.com/hughperkins/pytorch ~/pytorch
cd ~/pytorch
source ~/torch/install/bin/torch-activate
./build.sh
```
- clone this repo:
```
git clone https://github.com/hughperkins/pytorch-residual-networks ~/pytorch-residual-networks
cd ~/pytorch-residual-networks
```
- Download cifar dataset: `./download-cifar.sh`
- Run `python run.py`

## Possible issues, and how to deal with them

- Something about `no file './cunn.lua'`
  - reinstall cunn `luarocks install cunn`

## Changes

2016 April 12:
- working now :-)

2016 April 11:
- first forked from https://github.com/gcr/torch-residual-networks/network

