sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/CUB_small_joint.txt" -e "runs/error_CUB_small_joint.txt" run_cub.sh CUB_small joint
sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/CUB_blur_joint.txt" -e "runs/error_CUB_blur_joint.txt" run_cub.sh CUB_blur joint
 sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/CUB_tag_joint.txt" -e "runs/error_CUB_tag_joint.txt" run_cub.sh CUB_tag joint
