sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/CUB_small_independent.txt" -e "runs/error_CUB_small_independent.txt" run_cub.sh CUB_small independent
sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/CUB_blur_independent.txt" -e "runs/error_CUB_blur_independent.txt" run_cub.sh CUB_blur independent
 sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/CUB_tag_independent.txt" -e "runs/error_CUB_tag_independent.txt" run_cub.sh CUB_tag independent
