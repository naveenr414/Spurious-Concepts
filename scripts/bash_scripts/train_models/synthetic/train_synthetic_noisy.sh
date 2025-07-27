#!/bin/bash

seed=$1 

echo "Synthetic 1"
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small3 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small4 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small5 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small6 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_1_noisy --encoder_model small7 -epochs 50 -num_attributes 2 -num_classes 2 -seed $seed 

echo "Synthetic 2"
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small3 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small4 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small5 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small6 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2_noisy --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed 

echo "Synthetic 4"
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small3 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small4 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small5 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small6 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_4_noisy --encoder_model small7 -epochs 50 -num_attributes 8 -num_classes 2 -seed $seed 
