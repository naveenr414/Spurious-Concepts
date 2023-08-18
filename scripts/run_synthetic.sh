#!/bin/bash 

# Impact of model size
# for encoder_model in small3 small4 small5 small6 small7
# do 
#     python synthetic_experiments.py --num_objects 1 --encoder_model $encoder_model 
# done 
# for encoder_model in small3 small5 small7
# do 
#     python synthetic_experiments.py --num_objects 2 --encoder_model $encoder_model 
# done 
# for encoder_model in small3 small5 small7
# do
#     python synthetic_experiments.py --num_objects 4 --encoder_model $encoder_model 
# done 

# Impact of Noisy Dataset
echo "Impact of noisy dataset"
python synthetic_experiments.py --num_objects 2 --encoder_model inceptionv3 --noisy 

# Impact of weight decay 
echo "Impact of weight decay"
python synthetic_experiments.py --num_objects 2 --encoder_model inceptionv3 --weight_decay 0.004
python synthetic_experiments.py --num_objects 2 --encoder_model small3 --weight_decay 0.004

# Impact of optimizer 
for num_objects in 1 2 4 
do 
    echo "SAM $num_objects"
    python synthetic_experiments.py --num_objects $num_objects --encoder_model small7 --weight_decay 0.0004 --optimizer sam 
done 

