#!/bin/bash

python train_cbm.py -dataset CUB -model_type joint -num_attributes 112 -num_classes 200 -seed 42 -epochs 50 --encoder_model inceptionv3 -lr 0.001