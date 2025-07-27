#!/bin/bash

seed=$1

python scripts/notebooks/cub/cub_coco_intervention.py --seed $seed --num_concept_combinations 16 --dataset coco
python scripts/notebooks/cub/cub_coco_intervention.py --seed $seed --num_concept_combinations 32 --dataset coco 
python scripts/notebooks/cub/cub_coco_intervention.py --seed $seed --num_concept_combinations 48 --dataset coco
python scripts/notebooks/cub/cub_coco_intervention.py --seed $seed --num_concept_combinations 64 --dataset coco
