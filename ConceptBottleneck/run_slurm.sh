# sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/synthetic_43.txt" -e "runs/error_synthetic_43.txt" train_small_synthetic.sh 43
# sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/synthetic_44.txt" -e "runs/error_synthetic_44.txt" train_small_synthetic.sh 44

# sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=1:00:00 --gres=gpu:1 -N 1 -o "runs/cub_43.txt" -e "runs/error_cub_43.txt" train_small_cub.sh 43
# sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=1:00:00 --gres=gpu:1 -N 1 -o "runs/cub_44.txt" -e "runs/error_cub_44.txt" train_small_cub.sh 44

# sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/dsprites_43.txt" -e "runs/error_dsprites_43.txt" train_small_dsprites.sh 43
# sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/dsprites_44.txt" -e "runs/error_dsprites_44.txt" train_small_dsprites.sh 44

sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/independent_42.txt" -e "runs/error_indendent_42.txt" train_independent_models.sh 42
sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/independent_43.txt" -e "runs/error_indendent_43.txt" train_independent_models.sh 43
sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/independent_44.txt" -e "runs/error_indendent_44.txt" train_independent_models.sh 44