#!/bin/bash
#SBATCH -J ref5_eos_proper               # Job name
#SBATCH -o watch_folder/%A_%a.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=24G                     # server memory requested (per node)
#SBATCH -t 8:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=long               # Request partition
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:1                  # Type/number of GPUs needed
#SBATCH -c 6
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=1-5

# this is a bash array, with elements separated by spaces
seeds=( 1 2 3 4 5 )
seed=${seeds[$SLURM_ARRAY_TASK_ID-1]}

export NCCL_DEBUG=WARN
SCRATCH=/network/scratch/m/mohsin.hasan
#export HF_HOME=$SCRATCH/.cache/huggingface/
#export WORLD_SIZE=1

module --quiet load miniconda/3
conda activate /home/mila/m/mohsin.hasan/micromamba/envs/dplm
module --quiet load cuda/12.1.1
export TORCH_HOME=/home/mila/m/mohsin.hasan/scratch/torch_cache



python esm2_reference_experiment.py --seed ${seed} --num_particles 1 --num_seqs 5 --beta 50.0 --batch_num 10 --clamp_val -1.0
python esm2_reference_experiment.py --seed ${seed} --num_particles 5 --num_seqs 5 --beta 50.0 --batch_num 2 --clamp_val -1.0
python esm2_reference_experiment.py --seed ${seed} --num_particles 10 --num_seqs 5 --beta 50.0 --batch_num 1 --clamp_val -1.0
