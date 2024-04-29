#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --node-list=dill-sage

#This exits the script if any command fails

set -e

#SBATCH --exclude=ink-mia,ink-noah,glamor-ruby
#SBATCH --requeue
#SBATCH --qos=general