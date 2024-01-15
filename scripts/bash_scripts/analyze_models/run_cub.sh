#!/bin/bash 

seed=$1 

# python cub_analysis.py --seed $seed 
python cub_analysis.py --seed $seed --train_variation loss
python cub_analysis.py --seed $seed --train_variation half 
