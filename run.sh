#!/bin/bash

### do this type of thing for very interesting runs
# python3 train.py --num_routing_iterations=3 --dataset='cifar10' --gpu=0 --track --log_dir=''

### do this type of thing to record many runs
#python3 cnn_train.py --dataset='mnist' --gpu=0 &
#python3 cnn_train.py --dataset='mnist' --transform --gpu=1 &

#python3 train.py --dataset='fashion' --gpu=1 --transform &
#python3 train.py --dataset='cifar10' --gpu=2 --transform &
#python3 train.py --dataset='svhn' --gpu=3 --transform &



#python3 cnn_train.py --dataset='mnist'   --gpu=0 &
#python3 cnn_train.py --dataset='mnist'   --gpu=1 --transform &
#python3 cnn_train.py --dataset='fashion' --gpu=2 &
#python3 cnn_train.py --dataset='fashion' --gpu=3 --transform &

#python3 cnn_train.py --dataset='cifar10' --gpu=0 &
#python3 cnn_train.py --dataset='cifar10' --gpu=1 --transform &
python3 cnn_train.py --dataset='mnist'   --gpu=0             &
python3 cnn_train.py --dataset='mnist'   --gpu=1 --transform &


