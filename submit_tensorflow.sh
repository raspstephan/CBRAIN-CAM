#!/bin/sh
##
#SBATCH --account=glab             # The account name for the job.
#SBATCH --job-name=tensorflow_gpu   # The job name.
#SBATCH --gres=gpu:1               # Request 1 gpu (1-4 are valid).
# #SBATCH -c 2                      # The number of cpu cores to use.
#SBATCH --time=71:59:00            # The time the job will take to run.
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=pg2328@columbia.edu
# #SBATCH --exclusive
# #SBATCH --mem=120gb

module load intel-parallel-studio/2017
module load cuda80/toolkit cuda80/blas cudnn/5.1 
module load anaconda/2-4.2.0 

#mpiexec python ./main.py --epoch=50 --batch_size=1024 --lr=1e-3 --act=0 --input_names='TAP,QAP,OMEGA,SHFLX,LHFLX,LAT,dTdt_adiabatic,dQdt_adiabatic' --hidden='10,10' # > atoutfile
mpiexec python  ./main.py --run_validation=true --randomize=true --batch_size=4096 --optim=adam --lr=1e-3 --frac_train=0.8 --log_step=100 --epoch=50 --randomize=true --input_names='TAP,QAP,OMEGA,SHFLX,LHFLX,LAT,dTdt_adiabatic,dQdt_adiabatic' --hidden=32,32 --convo=false # > atoutfile
# mpiexec python  ./main.py --run_validation=true --randomize=true --batch_size=128 --optim=adam --lr=1e-3 --frac_train=0.8 --log_step=100 --epoch=50 --randomize=true --convo=true --input_names='TAP,QAP,OMEGA,SHFLX,LHFLX,LAT,dTdt_adiabatic,dQdt_adiabatic,QRL,QRS' --hidden=32,32 # > atoutfile
date

# End of script







