#!/bin/bash

#SBATCH --job-name="2d_7l_DNN"
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --mem 64GB
#SBATCH --gres gpu:1
#SBATCH --mail-user=chebert@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=7l_deconv_2in.out

module load python/2.7.5
module load tensorflow

basedir=/home/chebert/DonutNN/
scratchdir=/scratch/users/chebert

##restore=None
learningrate=.001
iters=5000
save=1
activation=None
mask=1
inputs=2

layers="[([4,4,2,96],2,'c'),([3,3,96,96],1,'c'),([4,4,96,96],2,'c'),([4,4,96,96],1,'d'),([4,4,96,96],1,'d'),([4,4,96,48],2,'d'),([3,3,48,1],1,'c')]"

#layers="[([6,6,1,48],1,'c'),([6,6,48,48],1,'c'),([6,6,48,48],1,'c'),([6,6,48,48],1,'c'),([6,6,48,96],1,'c'),([6,6,96,96],1,'c'),([6,6,96,96],1,'c'),([6,6,96,96],1,'c'),([6,6,96,96],1,'c'),([6,6,96,96],1,'c'),([6,6,96,96],1,'c'),([6,6,96,48],1,'c'),([3,3,48,1],1,'c')]"


command="python $basedir/DonutNet.py -f $scratchdir/simulatedData_plus.p -resdir $basedir/results/ -arch $layers -lr $learningrate -i $iters -a $activation -s $save -in $inputs"

echo $command
$command

