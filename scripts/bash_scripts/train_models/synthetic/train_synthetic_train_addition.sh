#!/bin/bash

seed=$1 

echo "Independent Models"
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed -model_type independent

echo "Adversarial Model"
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small3 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small4 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small5 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small6 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial 

echo "Adversarial Ablation"
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial --adversarial_weight 0.1
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial --adversarial_weight 0.25
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial --adversarial_weight 0.5
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial --adversarial_weight 1

python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial --adversarial_epsilon 0.01
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial --adversarial_epsilon 0.05
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial --adversarial_epsilon 0.1
python locality/cbm_variants/ConceptBottleneck/train_cbm.py -dataset synthetic_object/synthetic_2 --encoder_model small7 -epochs 50 -num_attributes 4 -num_classes 2 -seed $seed --train_variation adversarial --adversarial_epsilon 0.25
