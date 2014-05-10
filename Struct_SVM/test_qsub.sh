#!/bin/bash
#PBS -S /bin/bash
#PBS -N test
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
mpirun -np 96 --hostfile machinefile ./svm_struct_classify data_pascal/val.tst data_pascal/car.cad data_pascal/car_hard_0.mod data_pascal/val_hard_0.pre > data_pascal/car_test.log
