#!/bin/bash 

seed=$1

echo "CNN Models"
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model receptive_field3 --epochs 50 --num_objects 1 --seed $seed 
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model receptive_field4 --epochs 50 --num_objects 1 --seed $seed 
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model receptive_field5 --epochs 50 --num_objects 1 --seed $seed 
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model receptive_field6 --epochs 50 --num_objects 1 --seed $seed 
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model receptive_field7 --epochs 50 --num_objects 1 --seed $seed --lr 0.1 
