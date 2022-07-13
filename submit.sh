#!/bin/bash
#SBATCH --job-name                     torch
#SBATCH --partition                      gpu-normal
#SBATCH --nodes                           1
#SBATCH --tasks-per-node           6
#SBATCH --time                              72:00:00
#SBATCH --nodelist compute-9-10
#SBATCH --mem                             40G
#SBATCH --gres                               gpu:1
#SBATCH --constraint                    GTX1080
#SBATCH --output                          torch.%j.out
#SBATCH --error                             torch.%j.err
 
source /etc/profile
source /etc/profile.d/modules.sh
 
module add singularity/2.6.1
module add cuda/10.0.130
 
ulimit -s unlimited
 
# singularity exec --nv --bind /data:/data /share/apps/singularity/simg/pytorch/miniconda3-pytorch bash -c "source activate py36; which python;pip list;python -c 'import torch as t; print(t.cuda.is_available(),t.cuda.device_count())'"
singularity exec --nv --bind /data:/data /share/apps/singularity/simg/pytorch/miniconda3-pytorch bash -c "bash ./train.sh"
