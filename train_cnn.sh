#!/bin/bash

#SBATCH --job-name="c_noactiv_lr3_DNN"
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --mem 64GB
#SBATCH --gres gpu:1
#SBATCH --mail-user=chebert@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=dn_noactiv_conv_7l_lr3.out

module load python/2.7.5
module load tensorflow

basedir=/home/chebert/DonutNN/
scratchdir=/scratch/users/chebert

deconv=None
##restore=None
learningrate=.001
iters=5000
save=True
activation=None

command="python $basedir/DonutNet.py -f $scratchdir/simulatedData.p -resdir $basedir/results/ -d $deconv -lr $learningrate -i $iters -a $activation -s $save"

echo $command
$command

