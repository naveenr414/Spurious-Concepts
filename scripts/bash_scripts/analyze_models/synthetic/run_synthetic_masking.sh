#!/bin/bash 

seed=$1

python synthetic_masking.py --encoder_model small7 --epochs 50 --num_objects 1 --seed $seed 
python synthetic_masking.py --encoder_model small7 --epochs 50 --num_objects 2 --seed $seed 
python synthetic_masking.py --encoder_model small7 --epochs 50 --num_objects 4 --seed $seed 
python synthetic_masking.py --encoder_model small7 --epochs 50 --num_objects 8 --seed $seed 