#!/bin/bash

for i in 0 2 5 10 15 20 25
do 
    python train_cbm.py -dataset synthetic_2 --encoder_model mlp -epochs 50 -num_attributes 4 -num_classes 2 --expand_dim_encoder $i
done 