#!/bin/bash

seed=$1 

echo "Synthetic 2"
python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 --weight_decay 0.004 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 --weight_decay 0.04 -seed $seed 
