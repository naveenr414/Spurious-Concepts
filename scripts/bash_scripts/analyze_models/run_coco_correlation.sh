#!/bin/bash

seed=$1

python cub_correlation.py --seed $seed --num_concept_combinations 16 --dataset coco
python cub_correlation.py --seed $seed --num_concept_combinations 32 --dataset coco 
python cub_correlation.py --seed $seed --num_concept_combinations 48 --dataset coco
python cub_correlation.py --seed $seed --num_concept_combinations 64 --dataset coco
