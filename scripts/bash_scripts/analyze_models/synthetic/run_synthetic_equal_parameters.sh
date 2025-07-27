#!/bin/bash 

seed=$1

echo "CNN Models"
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model equal_parameter3 --epochs 50 --num_objects 1 --seed $seed 
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model equal_parameter4 --epochs 50 --num_objects 1 --seed $seed 
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model equal_parameter5 --epochs 50 --num_objects 1 --seed $seed 
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model equal_parameter6 --epochs 50 --num_objects 1 --seed $seed 
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model equal_parameter7 --epochs 50 --num_objects 1 --seed $seed --lr 0.1 
