#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # number of tasks per node
#SBATCH --gres=gpu:kepler:2          # for the gpu

#SBATCH --time=8:00:00               # time limits

#SBATCH --error=myJob.err            # standard error file
#SBATCH --output=myJob.out           # standard output file
#SBATCH --account=uTS18_BortolDL_0   # account name
#SBATCH --partition=gll_usr_gpuprod  # partition name


module load python
source my_venv/bin/activate
module load profile/deeplrn
module load cuda
module load cudnn

cd def
python train_def.py conf_20n_10lo_1k

deactivate
