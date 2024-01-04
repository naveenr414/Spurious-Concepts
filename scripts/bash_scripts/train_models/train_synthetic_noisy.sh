#!/bin/bash

seed=$1 

echo "Synthetic 1"
python train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small3 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small4 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small5 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small6 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small7 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 

echo "Synthetic 2"
python train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small3 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small4 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small5 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small6 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 

echo "Synthetic 4"
python train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small3 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small4 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small5 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small6 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small7 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
