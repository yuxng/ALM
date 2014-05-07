#!/bin/bash
#PBS -S /bin/bash
#PBS -N train
#PBS -A yuxiang
#PBS -l nodes=4:ppn=24
#PBS -l pmem=2000mb
#PBS -q cvglcuda
#PBS -M yuxiang@umich.edu	
#PBS -m abe
#PBS -j oe

echo "I ran on:"
cat $PBS_NODEFILE

# Change to the submission directory
cd $PBS_O_WORKDIR

# Perform tasks
mpirun -np 96 --hostfile machinefile ./svm_struct_learn -c 100000000 -l 0 -w 1 --l 100 --w 10 --f 1 --p 0 --h 0 data_pascal/car_wrap.dat data_pascal/car_unwrap.dat data_pascal/car_neg.dat data_pascal/car.cad > data_pascal/car_train.log
