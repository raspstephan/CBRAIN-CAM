#!/bin/sh
##
#SBATCH --account=glab             # The account name for the job.
#SBATCH --job-name=tensorflow_gpu   # The job name.
#SBATCH --gres=gpu:1               # Request 1 gpu (1-4 are valid).
# #SBATCH -c 2                      # The number of cpu cores to use.
#SBATCH --time=41:59:00            # The time the job will take to run.
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=pg2328@columbia.edu
# #SBATCH --exclusive
# #SBATCH --mem=120gb

module load intel-parallel-studio/2017
module load cuda80/toolkit cuda80/blas cudnn/5.1 
module load anaconda/2-4.2.0 

mpiexec python ./main.py --batch_size=4096 --dataset='SPDQ' --hidden='5' # > atoutfile

date

# End of script







