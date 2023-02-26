#!/bin/bash

CHEXPERT_PATH=../../chest_dataset/metadata_no_uncertainty/

python3 experiments.py chexpert Independent_CtoY --seed 42 -log_dir CHEXPERT/outputs/Independent -e 100 -optimizer sgd -use_attr -data_dir $CHEXPERT_PATH -n_attributes 13 -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 100 -expand_dim 128