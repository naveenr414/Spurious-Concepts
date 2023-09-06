#!/bin/bash

seed=$1 

echo "Training with 0 middle encoders"
python train_cbm.py -dataset synthetic_1 --encoder_model mlp -epochs 50 -num_attributes 2 -num_classes 2 --expand_dim_encoder 0 --num_middle_encoder 0 -seed $seed
 
echo "Training with 1 middle encoder" 
for i in 2 5 10 15 20 25
do 
    python train_cbm.py -dataset synthetic_1 --encoder_model mlp -epochs 50 -num_attributes 2 -num_classes 2 --expand_dim_encoder $i --num_middle_encoder 1 -seed $seed 
done 

echo "Training with 2 middle encoders" 
for i in 5 10 15
do 
    python train_cbm.py -dataset synthetic_1 --encoder_model mlp -epochs 50 -num_attributes 2 -num_classes 2 --expand_dim_encoder $i --num_middle_encoder 2 -seed $seed 
done 

echo "Training with 3 middle encoders" 
for i in 5 10 15
do 
    python train_cbm.py -dataset synthetic_1 --encoder_model mlp -epochs 50 -num_attributes 2 -num_classes 2 --expand_dim_encoder $i --num_middle_encoder 3 -seed $seed
done 