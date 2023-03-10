for model in sequential 
do
    for dataset in blur small tag
    do
        sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=1:00:00 --gres=gpu:1 -N 1 -o "runs/CUB_${dataset}_${model}.txt" -e "runs/error_CUB_${dataset}_${model}.txt" run_cub.sh CUB_${dataset} ${model}
    done
done
