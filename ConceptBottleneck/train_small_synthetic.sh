#!/bin/bash

seed=$1 

echo "Synthetic 1"
python train_cbm.py -dataset synthetic_1 --encoder_model small3 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_1 --encoder_model small4 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_1 --encoder_model small5 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_1 --encoder_model small6 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_1 --encoder_model small7 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 

python train_cbm.py -dataset synthetic_1 --encoder_model small3 -epochs 50 -num_attributes 2 -num_classes 2 --weight_decay 0.004 -seed $seed 

echo "Synthetic 2"
python train_cbm.py -dataset synthetic_2 --encoder_model small3 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_2 --encoder_model small5 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 

python train_cbm.py -dataset synthetic_2 --encoder_model small3 -epochs 50 -num_attributes 4 -num_classes 2 --weight_decay 0.004 -seed $seed 

echo "Synthetic 3"
python train_cbm.py -dataset synthetic_4 --encoder_model small3 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_4 --encoder_model small5 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_4 --encoder_model small7 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_4 --encoder_model small3 -epochs 50 -num_attributes 8 -num_classes 2 --weight_decay 0.004 -seed $seed 
