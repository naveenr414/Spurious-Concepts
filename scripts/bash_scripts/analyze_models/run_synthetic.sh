#!/bin/bash 

seed=$1

echo "CNN Models"
python synthetic_experiments.py --encoder_model small3 --epochs 50 --num_objects 1 --seed $seed 
python synthetic_experiments.py --encoder_model small4 --epochs 50 --num_objects 1 --seed $seed 
python synthetic_experiments.py --encoder_model small5 --epochs 50 --num_objects 1 --seed $seed 
python synthetic_experiments.py --encoder_model small6 --epochs 50 --num_objects 1 --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 50 --num_objects 1 --seed $seed 

python synthetic_experiments.py --encoder_model small3 --epochs 50 --num_objects 2 --seed $seed 
python synthetic_experiments.py --encoder_model small4 --epochs 50 --num_objects 2 --seed $seed 
python synthetic_experiments.py --encoder_model small5 --epochs 50 --num_objects 2 --seed $seed 
python synthetic_experiments.py --encoder_model small6 --epochs 50 --num_objects 2 --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 50 --num_objects 2 --seed $seed 

python synthetic_experiments.py --encoder_model small3 --epochs 50 --num_objects 4 --seed $seed 
python synthetic_experiments.py --encoder_model small4 --epochs 50 --num_objects 4 --seed $seed 
python synthetic_experiments.py --encoder_model small5 --epochs 50 --num_objects 4 --seed $seed 
python synthetic_experiments.py --encoder_model small6 --epochs 50 --num_objects 4 --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 50 --num_objects 4 --seed $seed 

python synthetic_experiments.py --encoder_model small3 --epochs 50 --num_objects 8 --seed $seed 
python synthetic_experiments.py --encoder_model small4 --epochs 50 --num_objects 8 --seed $seed 
python synthetic_experiments.py --encoder_model small5 --epochs 50 --num_objects 8 --seed $seed 
python synthetic_experiments.py --encoder_model small6 --epochs 50 --num_objects 8 --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 50 --num_objects 8 --seed $seed 

echo "MLP Models"
python synthetic_experiments.py --encoder_model mlp --epochs 50 --num_objects 1 --expand_dim_encoder 0 --num_middle_encoder 0 --seed $seed
 
for i in 5 10 15
do 
    python synthetic_experiments.py --encoder_model mlp --epochs 50 --num_objects 1 --expand_dim_encoder $i --num_middle_encoder 1 --seed $seed 
done 

for i in 5 10 15
do 
    python synthetic_experiments.py --encoder_model mlp --epochs 50 --num_objects 1 --expand_dim_encoder $i --num_middle_encoder 2 --seed $seed 
done 

for i in 5 10 15
do 
    python synthetic_experiments.py --encoder_model mlp --epochs 50 --num_objects 1 --expand_dim_encoder $i --num_middle_encoder 3 --seed $seed
done 

echo "Early Stopping"
python synthetic_experiments.py --encoder_model small7 --epochs 5 --num_objects 1  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 10 --num_objects 1  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 20 --num_objects 1  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 30 --num_objects 1  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 40 --num_objects 1  --seed $seed 

python synthetic_experiments.py --encoder_model small7 --epochs 5 --num_objects 2  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 10 --num_objects 2  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 20 --num_objects 2  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 30 --num_objects 2  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 40 --num_objects 2  --seed $seed 

python synthetic_experiments.py --encoder_model small7 --epochs 5 --num_objects 4  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 10 --num_objects 4  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 20 --num_objects 4  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 30 --num_objects 4  --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 40 --num_objects 4  --seed $seed 