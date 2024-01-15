#!/bin/bash 

seed=$1

for retrain_epochs in 0 5
do 
    for prune_rate in 0.25 0.5 0.75 0.95 
    do 
        echo ${retrain_epochs} ${prune_rate}
        python cub_pruning.py --encoder_model inceptionv3 --retrain_epochs ${retrain_epochs} --prune_rate ${prune_rate} --seed $seed  --pruning_technique weight
    done 

done

