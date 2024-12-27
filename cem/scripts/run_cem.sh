#!/bin/bash

experiment_name=$1
seed=$2
concept_loss_weight=$3
lr=$4
epochs=$5

num_gpus=1

LD_LIBRARY_PATH=../../anaconda3/lib python experiments/extract_cem_concepts.py --experiment_name $experiment_name --num_gpus $num_gpus --num_epochs ${epochs} --validation_epochs 25 --seed $seed --concept_pair_loss_weight 0 --concept_loss_weight $concept_loss_weight --lr $lr
