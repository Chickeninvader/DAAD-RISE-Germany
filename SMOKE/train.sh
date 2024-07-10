#!/bin/sh
#SBATCH --partition=compute # partition
#SBATCH --nodes=1  # number of nodes
#SBATCH --tasks-per-node=1  # number of tasks per node
#SBATCH --cpus-per-task 1  # number of cpus per task
#SBATCH --gres=gpu:1 # number of gpus (4 out of 8)
#SBATCH --mem=10000  # memory pool for all cores (in megabytes, if w/o suffix)
#SBATCH -t 0-00:01  # time (D-HH:MM)
#SBATCH -o /home/nvo/slurm_logs/slurm.%N.%j.out  # STDOUT
#SBATCH -e /home/nvo/slurm_logs/slurm.%N.%j.err  # STDERR
# show visible gpus
echo $CUDA_VISIBLE_DEVICES
# add path to micromamba
PATH=/usr/local/bin:$PATH
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate smoke
echo Starting Train
date
## < Here comes the command to be executed >
#python critical_classification/training_tensorflow.py
#python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
nvidia-smi
python setup.py build develop

echo Training complete
date