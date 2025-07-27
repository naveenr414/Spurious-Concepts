for num_objects in 2 4 8
do 
    for model in cem probcbm
    do 
        for seed in 42 43 44 
        do 
            python scripts/notebooks/synthetic/synthetic_leakage.py --model_type ${model} --encoder_model small7 --epochs 50 --num_objects ${num_objects} --seed $seed 
        done 
    done 
done