#!/bin/bash

python train_cbm.py -dataset synthetic_1 --encoder_model mlp -epochs 50 -num_attributes 2 -num_classes 2 --expand_dim_encoder 0 --num_middle_encoder 0
 

for i in 2 5 10 15 20 25
do 
    python train_cbm.py -dataset synthetic_1 --encoder_model mlp -epochs 50 -num_attributes 2 -num_classes 2 --expand_dim_encoder $i --num_middle_encoder 1 
done 

for i in 5 10 15
do 
    python train_cbm.py -dataset synthetic_1 --encoder_model mlp -epochs 50 -num_attributes 2 -num_classes 2 --expand_dim_encoder $i --num_middle_encoder 2
done 

for i in 5 10 15
do 
    python train_cbm.py -dataset synthetic_1 --encoder_model mlp -epochs 50 -num_attributes 2 -num_classes 2 --expand_dim_encoder $i --num_middle_encoder 3
done 