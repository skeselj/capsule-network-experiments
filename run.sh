#!/bin/bash

python3 main.py --batch_size_init=100 --num_routing_iterations=1 --gpu=2 &

python3 main.py --batch_size_init=100 --num_routing_iterations=2 --gpu=3 &

