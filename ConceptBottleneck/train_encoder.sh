#!/bin/bash

dataset=$1
nattributes=$2
nclasses=$3

python3 experiments.py cub Concept_XtoC --seed 42 -log_dir results/${dataset}/ -e 50 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir ../../cem/cem/${dataset}/preprocessed -n_attributes $nattributes -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 100 -bottleneck
