#!/bin/bash

seed=$1 

# echo "Freeze + Train"
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation half --scale_lr 1
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation half --scale_lr 2
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation half --scale_lr 3
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation half --scale_lr 4
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation half --scale_lr 5 


# echo "Concept Correlation"
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation loss --scale_factor 1
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation loss --scale_factor 1.25 
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation loss --scale_factor 1.5
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation loss --scale_factor 2
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation loss --scale_factor 3 

# echo "Independent Models"
# python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed -model_type independent

python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small3 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 
python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small4 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 
python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small5 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 
python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small6 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 
python train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 