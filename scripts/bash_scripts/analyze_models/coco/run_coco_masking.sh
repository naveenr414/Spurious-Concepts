#!/bin/bash 

seed=$1 

python scripts/notebooks/coco/coco_masking.py --seed $seed 
python scripts/notebooks/coco/coco_masking.py --seed $seed --model_type cem 
python scripts/notebooks/coco/coco_masking.py --seed $seed --model_type probcbm 