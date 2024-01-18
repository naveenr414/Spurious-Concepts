#!/bin/bash 

seed=$1

echo "CNN Models"
python synthetic_experiments.py --encoder_model small7 --epochs 50 --num_objects 2 --weight_decay 0.04 --seed $seed 
python synthetic_experiments.py --encoder_model small7 --epochs 50 --num_objects 2 --weight_decay 0.004 --seed $seed 