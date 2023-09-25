#!/bin/bash

seed=$1 

# python train_cbm.py -dataset synthetic_1 --encoder_model small7 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed -model_type independent
python train_cbm.py -dataset dsprites_20 -model_type independent -num_attributes 18 -num_classes 100 -seed $seed -epochs 50 --encoder_model small3 