#!/bin/bash

python3 train.py --num_routing_iterations=5 --dataset='mnist' --gpu=0 &
python3 train.py --num_routing_iterations=6 --dataset='mnist' --gpu=0 &
python3 train.py --num_routing_iterations=7 --dataset='mnist' --gpu=1 &
python3 train.py --num_routing_iterations=8 --dataset='mnist' --gpu=1 &


