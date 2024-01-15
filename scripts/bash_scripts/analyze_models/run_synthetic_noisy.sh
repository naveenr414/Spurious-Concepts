seed=$1

memory_used=$(nvidia-smi --query-gpu=memory.used --format csv | tail -n 1 | awk '{print $1}')
if ((memory_used > 500)); then 
    echo "GPU memory usage too high; kill some jobs first"
    exit 1
fi 


echo "CNN Models"
python synthetic_experiments.py --encoder_model small3 --epochs 50 --num_objects 1 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small4 --epochs 50 --num_objects 1 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small5 --epochs 50 --num_objects 1 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small6 --epochs 50 --num_objects 1 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small7 --epochs 50 --num_objects 1 --seed $seed --noisy

python synthetic_experiments.py --encoder_model small3 --epochs 50 --num_objects 2 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small4 --epochs 50 --num_objects 2 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small5 --epochs 50 --num_objects 2 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small6 --epochs 50 --num_objects 2 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small7 --epochs 50 --num_objects 2 --seed $seed --noisy

python synthetic_experiments.py --encoder_model small3 --epochs 50 --num_objects 4 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small4 --epochs 50 --num_objects 4 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small5 --epochs 50 --num_objects 4 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small6 --epochs 50 --num_objects 4 --seed $seed --noisy
python synthetic_experiments.py --encoder_model small7 --epochs 50 --num_objects 4 --seed $seed --noisy
