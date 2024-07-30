#!/bin/bash

ROOT_DIR=./..
CONFIG_DIR=${ROOT_DIR}/configs

config_file="${CONFIG_DIR}/eval/eval_olmo_configs.yaml"

sbatch_file="eval/eval_olmo.sh"

# Create a unique directory for this job's files
CACHE_JOB_DIR="cache_job/$(date +%Y%m%d_%H%M%S)"
mkdir -p ${CACHE_JOB_DIR}

# Copy the config file to the job directory
cp ${config_file} ${CACHE_JOB_DIR}/


# Submit the job, passing the job directory
sbatch --export=ALL,CACHE_JOB_DIR=${CACHE_JOB_DIR} ${sbatch_file}