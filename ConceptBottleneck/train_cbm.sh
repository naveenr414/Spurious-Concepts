#!/bin/bash

dataset=$1
model_type=$2
num_attributes=$3
num_classes=$4
seed=$5
epochs=$6

mkdir -p results/${dataset}

if [ "$model_type" = "independent" ]
then
    python3 experiments.py cub Independent_CtoY --seed $seed -log_dir results/${dataset}/${model_type}/bottleneck -e $epochs -use_attr -optimizer sgd  -data_dir ../../cem/cem/${dataset}/preprocessed -n_attributes $num_attributes -no_img -b 64 -weight_decay 0.00005 -lr 0.01 -scheduler_step 100 -num_classes $num_classes
    python3 experiments.py cub Concept_XtoC --seed $seed -ckpt 1 -log_dir results/${dataset}/${model_type}/concept -e $epochs -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir ../../cem/cem/${dataset}/preprocessed -n_attributes $num_attributes -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -num_classes $num_classes -scheduler_step 100 -bottleneck
elif [ "$model_type" = "sequential" ]
then 
    python3 experiments.py cub Concept_XtoC --seed $seed -ckpt 1 -log_dir results/${dataset}/${model_type}/concept -e $epochs -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir ../../cem/cem/${dataset}/preprocessed -n_attributes $num_attributes -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 100 -num_classes $num_classes -bottleneck
    python3 CUB/generate_new_data.py ExtractConcepts --model_path results/${dataset}/sequential/concept/best_model_42.pth --data_dir ../../cem/cem/${dataset}/preprocessed -num_classes $num_classes --out_dir results/${dataset}/sequential/output/
    python3 experiments.py cub Sequential_CtoY --seed $seed -log_dir results/${dataset}/${model_type}/bottleneck -e $epochs -use_attr -optimizer sgd  -data_dir results/${dataset}/sequential/output -n_attributes $num_attributes -num_classes $num_classes -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 100
elif [ "$model_type" = "joint" ]
then
        python3 experiments.py cub Joint --seed $seed -ckpt 1 -log_dir results/${dataset}/${model_type} -e $epochs -optimizer sgd -pretrained -use_aux  -weighted_loss multiple -use_attr -data_dir ../../cem/cem/${dataset}/preprocessed -n_attributes $num_attributes -attr_loss_weight 1.0 -normalize_loss -b 64 -weight_decay 0.0004 -num_classes $num_classes -lr 0.05 -scheduler_step 30 -end2end -use_sigmoid
else 
    echo "Model type ${model_type} not found"
fi

