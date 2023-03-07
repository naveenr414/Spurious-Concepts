#!/bin/bash

dataset=$1
model_type=$2

if [ "$model_type" = "independent" ]
then
    python3 experiments.py cub Concept_XtoC --seed 42 -ckpt 1 -log_dir results/${dataset}/${model_type}/concept -e 50 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir ../../cem/cem/${dataset}/preprocessed -n_attributes 113 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 100 -bottleneck
    python3 experiments.py cub Independent_CtoY --seed 42 -log_dir results/${dataset}/${model_type}/bottleneck -e 50 -use_attr -optimizer sgd  -data_dir ../../cem/cem/${dataset}/preprocessed -n_attributes 113 -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 100
elif [ "$model_type" = "joint" ]
then
        python3 experiments.py cub Joint --seed 42 -ckpt 1 -log_dir results/${dataset}/${model_type} -e 50 -optimizer sgd -pretrained -use_aux  -weighted_loss multiple -use_attr -data_dir ../../cem/cem/${dataset}/preprocessed -n_attributes 113 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 100 -end2end 
else 
    echo "Model type ${model_type} not found"
fi

