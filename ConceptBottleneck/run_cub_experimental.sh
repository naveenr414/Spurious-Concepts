#!/bin/bash

dataset=$1
train_addition=$2
model_type=joint

mkdir other_results/${dataset}/${model_type}_${train_addition} 
python3 experiments.py cub Joint --seed 42 -ckpt 1 -log_dir other_results/${dataset}/${model_type}_${train_addition} -e 50 -optimizer sgd -pretrained -use_aux  -weighted_loss multiple -use_attr -data_dir ../../cem/cem/${dataset}/preprocessed -n_attributes 113 -attr_loss_weight .001 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 100 -train_addition ${train_addition} -end2end 
