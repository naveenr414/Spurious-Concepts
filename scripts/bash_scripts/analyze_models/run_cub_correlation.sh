#!/bin/bash

seed=$1

python cub_correlation.py --seed $seed --num_concept_combinations 50 --dataset CUB
python cub_correlation.py --seed $seed --num_concept_combinations 100 --dataset CUB 
python cub_correlation.py --seed $seed --num_concept_combinations 150 --dataset CUB
python cub_correlation.py --seed $seed --num_concept_combinations 200 --dataset CUB
