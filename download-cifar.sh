#!/bin/bash

CIFARMD5=$(md5sum cifar-10-python.tar.gz | awk '{print $1}')
if [[ $CIFARMD5 != c58f30108f718f92721af3b95e74349a ]]; then {
  wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar-10-python.tar.gz
} fi

if [[ ! -d cifar-10-batches-py ]]; then {
  tar -xf cifar-10-python.tar.gz
} fi

