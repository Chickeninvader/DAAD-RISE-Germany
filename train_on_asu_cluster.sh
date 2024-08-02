#!/bin/bash

# Reference: https://asurc.atlassian.net/wiki/spaces/RC/pages/1908998178/Sol+Hardware+-+How+to+Request
#SBATCH -N 1            # number of nodes
#SBATCH -c 2            # number of cores
#SBATCH -G a100:1            # Request 1 gpu
#SBATCH -t 1-0:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -o /home/ngocbach/slurm_logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e /home/ngocbach/slurm_logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required software
module load mamba/latest

# Activate our enviornment
source activate daad7

# Change to the directory of our script
cd /scratch/ngocbach/DAAD-RISE-Germany

echo Starting Train
date
## < Here comes the command to be executed >

# For training the model
#python critical_classification/training_torch.py \
#  --model_name Swin3D \
#  --image_batch_size 10 \
#  --additional_config train_from_scratch

# For YOLOv1_video
#  --additional_config no_fc
# For Swin3D:
#  --additional_config train_from_scratch

# For visualization

python critical_classification/inference.py \
  --model_name Swin3D \
  --data_location /data/nvo/ \
  --pretrained_path critical_classification/save_models/Dall_MSwin3D_lr0.0001_lossBCE_e20_scosine_Aexperiment_20240729_053945/Dall_MSwin3D_lr0.0001_lossBCE_e20_scosine_Aexperiment_20240729_053945_e14_best.pth \
  --infer_all_video


# For other purpose
# download dataset
#python critical_classification/src/dataset/download_video.py

echo Training complete
date