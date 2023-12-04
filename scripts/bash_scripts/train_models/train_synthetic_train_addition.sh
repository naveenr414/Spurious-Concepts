#!/bin/bash

seed=$1 

echo "Freeze + Train"
python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation half --scale_lr 4 --debugging 

echo "Concept Correlation"
python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation loss --scale_factor 2 --debugging 

echo "Independent Models"
python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed -model_type independent --debugging

