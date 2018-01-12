#!/bin/bash

python capsule_network --gpu=1 --batch_size_init=100 --num_routing_iterations=1
python capsule_network --batch_size_init=100 --num_routing_iterations=2
python capsule_network --batch_size_init=100 --num_routing_iterations=4
python capsule_network --batch_size_init=100 --num_routing_iterations=5

python capsule_network --batch_size_init=25 --num_routing_iterations=3
python capsule_network --batch_size_init=50 --num_routing_iterations=3
python capsule_network --batch_size_init=200 --num_routing_iterations=3
python capsule_network --batch_size_init=400 --num_routing_iterations=3

