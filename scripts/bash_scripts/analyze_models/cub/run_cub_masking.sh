#!/bin/bash 

seed=$1 

python cub_masking.py --seed $seed 
python cub_masking.py --seed $seed --model_type cem 
python cub_masking.py --seed $seed --model_type probcbm 
python cub_label_free.py --seed $seed