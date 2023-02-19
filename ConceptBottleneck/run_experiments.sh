CUB_path=$1

python3 experiments.py cub Concept_XtoC --seed 42 -ckpt 1 -log_dir CUB/outputs -e 1 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir $CUB_path -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck
