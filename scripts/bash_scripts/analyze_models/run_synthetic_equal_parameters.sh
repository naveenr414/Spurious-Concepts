#!/bin/bash 

seed=$1

echo "CNN Models"
python synthetic_experiments.py --encoder_model equal_parameter3 --epochs 50 --num_objects 1 --seed $seed 
python synthetic_experiments.py --encoder_model equal_parameter4 --epochs 50 --num_objects 1 --seed $seed 
python synthetic_experiments.py --encoder_model equal_parameter5 --epochs 50 --num_objects 1 --seed $seed 
python synthetic_experiments.py --encoder_model equal_parameter6 --epochs 50 --num_objects 1 --seed $seed 
python synthetic_experiments.py --encoder_model equal_parameter7 --epochs 50 --num_objects 1 --seed $seed 
