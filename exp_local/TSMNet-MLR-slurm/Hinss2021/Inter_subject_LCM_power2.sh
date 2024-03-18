#!/bin/bash

#SBATCH -p lambda
#SBATCH -N 1
#SBATCH --mem-per-cpu=10G
#SBATCH -c 8
#SBATCH -A staff
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -o ../../outputs/slurm_log/output_%j.txt

# lambda gpupart chaos
# staff staff shared-mhug-staff
source /nfs/data_todi/zchen/anaconda/bin/activate
conda activate spdnet
cd /nfs/data_todi/zchen/Realeased_code/SPDMLR

## LCM-POWER
python TSMNet-MLR.py -m\
  data_dir=/nfs/data_todi/zchen/Datasets/mne_data\
  evaluation=inter-subject+uda\
  nnet.model.classifier=SPDMLR\
  nnet.model.metric=SPDLogCholeskyMetric\
  nnet.model.power=-1.,-0.75,-0.5,-0.25,1.,0.75,0.5,0.25\
  hydra.job_logging.handlers.file.filename=LCM-POWER2.log