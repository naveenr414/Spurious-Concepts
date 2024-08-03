for num_objects in 8 # 2 4 
do 
    for model in cem probcbm
    do 
        for seed in 42 43 44 
        do 
            python synthetic_experiments.py --model_type ${model} --encoder_model small7 --epochs 50 --num_objects ${num_objects} --seed $seed 
        done 
    done 
done