#!/bin/bash

seed=$1 

# echo "Freeze + Train"
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation half --scale_lr 1
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation half --scale_lr 2
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation half --scale_lr 3
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation half --scale_lr 4
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation half --scale_lr 5 


# echo "Concept Correlation"
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation loss --scale_factor 1
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation loss --scale_factor 1.25 
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation loss --scale_factor 1.5
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation loss --scale_factor 2
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation loss --scale_factor 3 

# echo "Independent Models"
# python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --model_type independent

python synthetic_experiments.py --num_objects 2 --encoder_model small3 --epochs 50 --seed $seed --train_variation adversarial
python synthetic_experiments.py --num_objects 2 --encoder_model small4 --epochs 50 --seed $seed --train_variation adversarial
python synthetic_experiments.py --num_objects 2 --encoder_model small5 --epochs 50 --seed $seed --train_variation adversarial
python synthetic_experiments.py --num_objects 2 --encoder_model small6 --epochs 50 --seed $seed --train_variation adversarial
python synthetic_experiments.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial
