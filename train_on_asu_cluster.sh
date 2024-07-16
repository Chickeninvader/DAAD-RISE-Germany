#!/bin/bash

# Reference: https://asurc.atlassian.net/wiki/spaces/RC/pages/1908998178/Sol+Hardware+-+How+to+Request
#SBATCH -N 1            # number of nodes
#SBATCH -c 2            # number of cores
#SBATCH -G 1            # Request 1 gpu
#SBATCH -t 0-0:30:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required software
module load mamba/latest

# Activate our enviornment
source activate test1

# Change to the directory of our script
cd /scratch/ngocbach/DAAD-RISE-Germany

#Run the software/python script
python critical_classification/training_tensorflow.py