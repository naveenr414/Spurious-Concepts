#!/bin/bash

seed=$1 

# Main Run
python train_cbm.py -dataset coco --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 15 -num_classes 2 -seed $seed --attr_loss_weight 0.05 --optimizer adam --scheduler_step 100 -lr 5e-4

# Train Variations
python train_cbm.py -dataset coco --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 15 -num_classes 2 -seed 42 --attr_loss_weight 0.05 --optimizer adam --scheduler_step 100 -lr 5e-4 --train_variation half --scale_lr 5 
python train_cbm.py -dataset coco --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 15 -num_classes 2 -seed 42 --attr_loss_weight 0.05 --optimizer adam --scheduler_step 100 -lr 5e-4 --train_variation loss --scale_factor 1.25 