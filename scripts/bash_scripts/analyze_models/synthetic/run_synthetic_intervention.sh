#!/bin/bash

seed=$1

python scripts/notebooks/synthetic/synthetic_intervention.py --seed $seed --encoder_model small7 --num_concept_combinations 1 --num_objects 4
python scripts/notebooks/synthetic/synthetic_intervention.py --seed $seed --encoder_model small7 --num_concept_combinations 2 --num_objects 4
python scripts/notebooks/synthetic/synthetic_intervention.py --seed $seed --encoder_model small7 --num_concept_combinations 4 --num_objects 4
python scripts/notebooks/synthetic/synthetic_intervention.py --seed $seed --encoder_model small7 --num_concept_combinations 8 --num_objects 4
python scripts/notebooks/synthetic/synthetic_intervention.py --seed $seed --encoder_model small7 --num_concept_combinations 12 --num_objects 4
python scripts/notebooks/synthetic/synthetic_intervention.py --seed $seed --encoder_model small7 --num_concept_combinations 15 --num_objects 4