#!/bin/bash
#SBATCH -J tsf_3img    # Name that will show up in squeueu
#SBATCH --gres=gpu:1         # Request 4 GPU "generic resource"
#SBATCH --time=4-00:00       # Max job time is 3 hours
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=long     # long partition (allows up to 7 days runtime)
#SBATCH --nodelist=cs-venus-06   # if needed, set the node you want (similar to -w xyz)
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6

# Your experiment setup logic here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfuse
hostname
echo $CUDA_AVAILABLE_DEVICES
export OMP_NUM_THREADS=1

# Note the actual command is run through srun
srun python -u transfuser/train.py 
