#!/bin/bash

python3 capsule_network.py --batch_size_init=100 --num_routing_iterations=1 --gpu=2 &

python3 capsule_network.py --batch_size_init=100 --num_routing_iterations=2 --gpu=3 &
python3 capsule_network.py --batch_size_init=100 --num_routing_iterations=4 --gpu=3 &
