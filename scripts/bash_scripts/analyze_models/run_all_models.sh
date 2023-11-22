#!/bin/bash

# Synthetic Models
sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=1:00:00 --mem=16g -o "runs/synthetic_42.txt" -e "runs/error_synthetic_42.txt" run_synthetic.sh 42
sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=1:00:00 --mem=16g -o "runs/synthetic_43.txt" -e "runs/error_synthetic_43.txt" run_synthetic.sh 43
sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=1:00:00 --mem=16g -o "runs/synthetic_44.txt" -e "runs/error_synthetic_44.txt" run_synthetic.sh 44

# CUB Models 
sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=2:00:00 --mem=16g  -o "runs/cub_42.txt" -e "runs/error_cub_42.txt" run_cub.sh 42
sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=2:00:00 --mem=32g -o "runs/cub_43.txt" -e "runs/error_cub_43.txt" run_cub.sh 43
sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=2:00:00 --mem=32g -o "runs/cub_44.txt" -e "runs/error_cub_44.txt" run_cub.sh 44

# DSprites Models 
sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=1:00:00 --mem=16g -o "runs/dsprites_42.txt" -e "runs/error_dsprites_42.txt" run_adversarial_concepts.sh 42
sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=1:00:00 --mem=16g -o "runs/dsprites_43.txt" -e "runs/error_dsprites_43.txt" run_adversarial_concepts.sh 43
sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=1:00:00 --mem=16g -o "runs/dsprites_44.txt" -e "runs/error_dsprites_44.txt" run_adversarial_concepts.sh 44
