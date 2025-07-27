#!/bin/bash

seed=$1 

echo "Synthetic 1"
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model small7 -epochs 5 -num_attributes 2 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model small7 -epochs 10 -num_attributes 2 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model small7 -epochs 20 -num_attributes 2 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model small7 -epochs 30 -num_attributes 2 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1 --encoder_model small7 -epochs 40 -num_attributes 2 -num_classes 2 -seed $seed 

echo "Synthetic 2"
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 5 -num_attributes 4 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 10 -num_attributes 4 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 20 -num_attributes 4 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 30 -num_attributes 4 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 40 -num_attributes 4 -num_classes 2 -seed $seed 

echo "Synthetic 4"
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4 --encoder_model small7 -epochs 5 -num_attributes 8 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4 --encoder_model small7 -epochs 10 -num_attributes 8 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4 --encoder_model small7 -epochs 20 -num_attributes 8 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4 --encoder_model small7 -epochs 30 -num_attributes 8 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4 --encoder_model small7 -epochs 40 -num_attributes 8 -num_classes 2 -seed $seed 