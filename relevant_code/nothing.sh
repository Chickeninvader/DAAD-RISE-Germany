#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores
#SBATCH -t 0-01:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=someone@asu.edu
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate scicomp

##
# Generate 1,000,000 random numbers in bash,
#   then store sorted results
#   in `Distribution.txt`
##
for i in $(seq 1 1e6); do
  printf "%d\n" $RANDOM
done | sort -n > Distribution.txt
# Plot Histogram using python and a heredoc
python << EOF
import pandas as pd, seaborn as sns
sns.mpl.use('Agg')
sns.set(color_codes=True)
df = pd.read_csv('Distribution.txt',header=None,names=['rand'])
sns.distplot(df,kde=False,rug=True)
sns.mpl.pyplot.xlim(0,df['rand'].max())
sns.mpl.pyplot.xlabel('Integer Result')
sns.mpl.pyplot.ylabel('Count')
sns.mpl.pyplot.title('Sampled Uniform Distribution')
sns.mpl.pyplot.savefig('Histogram.png')
EOF
