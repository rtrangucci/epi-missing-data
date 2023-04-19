#!/bin/sh

#!/bin/bash
#SBATCH --job-name simu-mod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=2:00:00
#SBATCH --account=jzelner1
#SBATCH --partition=standard
#SBATCH --mail-type=NONE
#SBATCH --export=ALL
#SBATCH --output=%x-%j.log
#SBATCH --array=1-200

i=$SLURM_ARRAY_TASK_ID
cd $SLURM_SUBMIT_DIR
Rscript --verbose run-mod-server.R "data_10p.RDS" $i "cmdstan_model_full.RDS" "simu-mod-10p" "vars_to_pull_full_mod.RDS"
# Run rscript on each node
