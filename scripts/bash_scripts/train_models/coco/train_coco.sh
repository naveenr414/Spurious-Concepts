#!/bin/bash

seed=$1 

# Main Run
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset coco --encoder_model inceptionv3 --pretrained -epochs 25 -num_attributes 10 -num_classes 2 -seed $seed --attr_loss_weight 0.1 --optimizer adam --scheduler_step 100 -lr 0.005