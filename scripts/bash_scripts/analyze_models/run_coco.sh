#!/bin/bash 

seed=$1 

python coco_analysis.py --seed $seed 
python coco_analysis.py --seed $seed --train_variation half --scale_lr 5   
python coco_analysis.py --seed $seed --train_variation loss --scale_factor 1.25
