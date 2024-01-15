#!/bin/bash

seed=$1 

# Main Run
# python train_cbm.py -dataset CUB --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 112 -num_classes 200 -seed $seed --attr_loss_weight 0.01 --optimizer adam --scheduler_step 100 -lr 0.005

memory_used=$(nvidia-smi --query-gpu=memory.used --format csv | tail -n 1 | awk '{print $1}')
if ((memory_used > 500)); then 
    echo "GPU memory usage too high; kill some jobs first"
    exit 1
fi 


# Train Variations
python train_cbm.py -dataset CUB --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 112 -num_classes 200 -seed $seed --attr_loss_weight 0.01 --optimizer adam --scheduler_step 100 -lr 0.005 --train_variation half --scale_lr 5 
sleep 300
python train_cbm.py -dataset CUB --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 112 -num_classes 200 -seed $seed --attr_loss_weight 0.01 --optimizer adam --scheduler_step 100 -lr 0.005 --train_variation loss --scale_factor 1.25 
sleep 300