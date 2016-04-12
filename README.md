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
pushd ~/torch
bash install-deps
./install.sh
popd
```
- install torch cudnn and nninit:
```
luarocks install cudnn
luarocks install nninit
```
- Setup python (tested on 2.7 for now; 3.4 will follow):
```
sudo apt-get install python2.7-dev
virtualenv -p python2.7 env27
source env27/bin/activate
pip install docopt
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

2016 April 17:
- working now :-)

2016 April 16:
- first forked from https://github.com/gcr/torch-residual-networks/network

