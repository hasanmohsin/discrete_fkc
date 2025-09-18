#!/bin/bash

# Example Bash script for running an unconditional experiment

# Set experiment parameters
EXPERIMENT_NAME="uncond_exp"
OUTPUT_DIR="./results/${EXPERIMENT_NAME}"
LOG_FILE="${OUTPUT_DIR}/run.log"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run the experiment (replace with your actual command)
echo "Running unconditional experiment..." | tee "${LOG_FILE}"
python run_experiment.py  --seed 1 --seq_length 50 --num_particles 5 --num_seqs 5 --beta 200.0 | tee -a "${LOG_FILE}"

echo "Experiment finished. Results saved in ${OUTPUT_DIR}"