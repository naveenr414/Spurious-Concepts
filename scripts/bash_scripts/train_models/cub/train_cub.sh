#!/bin/bash

seed=$1 

# Main Run
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset CUB --encoder_model inceptionv3 --pretrained -epochs 100 -num_attributes 112 -num_classes 200 -seed $seed --attr_loss_weight 0.01 --optimizer adam --scheduler_step 100 -lr 0.005