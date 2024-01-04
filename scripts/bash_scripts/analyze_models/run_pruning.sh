#!/bin/bash 

seed=$1

echo "CNN Models"

for num_objects in 2 4
do 
    for encoder_model in small3 small4 small5 small6 small7 
    do 
        for retrain_epochs in 0 5
        do 
            for prune_rate in 0.25 0.5 0.75 0.95 
            do 
                echo ${encoder_model} ${retrain_epochs} ${prune_rate} ${num_objects}
                python pruning.py --encoder_model ${encoder_model} --retrain_epochs ${retrain_epochs} --prune_rate ${prune_rate} --seed $seed  --pruning_technique weight --num_objects $num_objects --dataset_name synthetic_object/synthetic_${num_objects}
            done 
        done 
    done 

    # for hidden_layers in 5 10 15 
    # do 
    #     for retrain_epochs in 0 5
    #     do 
    #         for prune_rate in 0.25 0.5 0.75 0.95 
    #         do 
    #             echo ${hidden_layers} ${retrain_epochs} ${prune_rate} ${num_objects}
    #             python pruning.py --encoder_model mlp --num_middle_encoder 1 --expand_dim_encoder ${hidden_layers} --retrain_epochs ${retrain_epochs} --prune_rate ${prune_rate} --seed $seed  --pruning_technique weight --num_objects $num_objects --dataset_name synthetic_object/synthetic_${num_objects}
    #         done 
    #     done 
    # done 

done 
