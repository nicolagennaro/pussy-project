#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # 2 tasks per node
#SBATCH --gres=gpu:kepler:1          # for the gpu

#SBATCH --time=4:00:00               # time limits: 4 hour

#SBATCH --error=myJob.err            # standard error file
#SBATCH --output=myJob.out           # standard output file
#SBATCH --account=uTS18_BortolDL_0   # account name
#SBATCH --partition=gll_usr_gpuprod  # partition name


module load python
source my_venv/bin/activate
module load profile/deeplrn
module load cuda
module load cudnn

python train_def.py configuration_alpha.txt

deactivate
