#!/bin/bash

seed=$1

echo "Synthetic 1"
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model receptive_field3 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model receptive_field4 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model receptive_field5 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model receptive_field6 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model receptive_field7 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed -lr 0.1
