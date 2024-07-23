#!/bin/sh
#SBATCH --partition=compute # partition
#SBATCH --nodes=1  # number of nodes
#SBATCH --tasks-per-node=1  # number of tasks per node
#SBATCH --cpus-per-task 1  # number of cpus per task
#SBATCH --gres=gpu:1 # number of gpus (4 out of 8)
#SBATCH --mem=10000  # memory pool for all cores (in megabytes, if w/o suffix)
#SBATCH -t 0-18:00  # time (D-HH:MM)
#SBATCH -o /home/nvo/slurm_logs/slurm.%N.%j.out  # STDOUT
#SBATCH -e /home/nvo/slurm_logs/slurm.%N.%j.err  # STDERR
# show visible gpus
echo $CUDA_VISIBLE_DEVICES
# add path to micromamba
PATH=/usr/local/bin:$PATH
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate daad_torch_2
echo Starting Train
date
## < Here comes the command to be executed >

# For training the model
python critical_classification/training_torch.py \
  --model_name YOLOv1_video \
  --data_location /data/nvo/

# For visualization

#python critical_classification/inference.py \
#  --model_name Swin3D \
#  --data_location /data/nvo/ \
#  --pretrained_path critical_classification/save_models/Dall_MSwin3D_lr0.0001_lossBCE_e40_scosine_Aexperiment_20240722_173158.pth \
#  --infer_all_video

# For other purpose
# download dataset
#python critical_classification/src/dataset/download_video.py

echo Training complete
date