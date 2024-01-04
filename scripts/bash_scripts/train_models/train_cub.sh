#!/bin/bash

seed=$1 

# Main Run
python train_cbm.py -dataset CUB --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 112 -num_classes 200 -seed $seed --attr_loss_weight 0.01 --optimizer adam --scheduler_step 100 -lr 0.005

# Train Variations
python train_cbm.py -dataset CUB --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 112 -num_classes 200 -seed $seed --attr_loss_weight 0.01 --optimizer adam --scheduler_step 100 -lr 0.005 --train_variation half --scale_lr 5 
python train_cbm.py -dataset CUB --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 112 -num_classes 200 -seed $seed --attr_loss_weight 0.01 --optimizer adam --scheduler_step 100 -lr 0.005 --train_variation loss --scale_factor 1.25 