#!/bin/bash 

seed=$1 

python train_cbm.py -dataset dsprites_5 -model_type joint -num_attributes 18 -num_classes 100 -seed $seed -epochs 50 --encoder_model small3
python train_cbm.py -dataset dsprites -model_type joint -num_attributes 18 -num_classes 100 -seed $seed -epochs 50 --encoder_model small3
python train_cbm.py -dataset dsprites_15 -model_type joint -num_attributes 18 -num_classes 100 -seed $seed -epochs 50 --encoder_model small3
python train_cbm.py -dataset dsprites_20 -model_type joint -num_attributes 18 -num_classes 100 -seed $seed -epochs 50 --encoder_model small3
