seed=$1

echo "CNN Models"
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small3 --epochs 50 --num_objects 1 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small4 --epochs 50 --num_objects 1 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small5 --epochs 50 --num_objects 1 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small6 --epochs 50 --num_objects 1 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small7 --epochs 50 --num_objects 1 --seed $seed --noisy

python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small3 --epochs 50 --num_objects 2 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small4 --epochs 50 --num_objects 2 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small5 --epochs 50 --num_objects 2 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small6 --epochs 50 --num_objects 2 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small7 --epochs 50 --num_objects 2 --seed $seed --noisy

python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small3 --epochs 50 --num_objects 4 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small4 --epochs 50 --num_objects 4 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small5 --epochs 50 --num_objects 4 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small6 --epochs 50 --num_objects 4 --seed $seed --noisy
python scripts/notebooks/synthetic/synthetic_leakage.py --encoder_model small7 --epochs 50 --num_objects 4 --seed $seed --noisy
