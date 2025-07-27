#!/bin/bash

seed=$1 

echo "Adversarial training" 

python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small3 --epochs 50 --seed $seed --train_variation adversarial
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small4 --epochs 50 --seed $seed --train_variation adversarial
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small5 --epochs 50 --seed $seed --train_variation adversarial
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small6 --epochs 50 --seed $seed --train_variation adversarial
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial

python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial --adversarial_weight 0.1
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial --adversarial_weight 0.25
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial --adversarial_weight 0.5
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial --adversarial_weight 1

python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial --adversarial_epsilon 0.01
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial --adversarial_epsilon 0.05
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial --adversarial_epsilon 0.1
python scripts/notebooks/synthetic/synthetic_leakage.py --num_objects 2 --encoder_model small7 --epochs 50 --seed $seed --train_variation adversarial --adversarial_epsilon 0.25
