#!/bin/bash

seed=$1

echo "Synthetic 1"
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model equal_parameter3 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model equal_parameter4 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model equal_parameter5 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model equal_parameter6 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model equal_parameter7 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed -lr 0.1
