#!/bin/bash

seed=$1 

python train_cbm.py -dataset CUB -model_type joint -num_attributes 112 -num_classes 200 -seed $seed -epochs 50 --encoder_model small3 -lr 0.1
python train_cbm.py -dataset CUB -model_type joint -num_attributes 112 -num_classes 200 -seed $seed -epochs 50 --encoder_model small7 -lr 0.1