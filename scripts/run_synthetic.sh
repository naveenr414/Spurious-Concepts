#!/bin/bash 

# Impact of model size
# echo "1 Object"
# for encoder_model in small3 small4 small5 small6 small7
# do 
#     python synthetic_experiments.py --num_objects 1 --encoder_model $encoder_model 
# done 

echo "2 Objects" 
for encoder_model in small3 small5 small7
do 
    python synthetic_experiments.py --num_objects 2 --encoder_model $encoder_model 
done

echo "4 Objects"
for encoder_model in small3 small5 small7
do
    python synthetic_experiments.py --num_objects 4 --encoder_model $encoder_model 
done 

# Impact of Noisy Dataset
echo "Impact of noisy dataset"
python synthetic_experiments.py --num_objects 2 --encoder_model inceptionv3 --noisy 

# Impact of weight decay 
echo "Impact of weight decay"
python synthetic_experiments.py --num_objects 2 --encoder_model inceptionv3 --weight_decay 0.004

# Impact of optimizer 
python synthetic_experiments.py --num_objects 4 --encoder_model small3 --weight_decay 0.0004 --optimizer sam 
python synthetic_experiments.py --num_objects 4 --encoder_model small7 --weight_decay 0.0004 --optimizer sam 


# MLP Models 
python synthetic_experiments.py --num_objects 1 --encoder_model mlp_0_0 --weight_decay 0.0004

for i in 2 5 10 15 20 25
do 
    python synthetic_experiments.py --num_objects 1 --encoder_model mlp_${i}_1 --weight_decay 0.0004
done 

for i in 5 10 15
do 
    python synthetic_experiments.py --num_objects 1 --encoder_model mlp_${i}_2 --weight_decay 0.0004
done 

for i in 5 10 15
do 
    python synthetic_experiments.py --num_objects 1 --encoder_model mlp_${i}_3 --weight_decay 0.0004
done 