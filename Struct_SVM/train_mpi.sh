#!/bin/bash

CLS=car
DATADIR=../Main/data_aspect/

# Perform tasks
cat train_mpi.sh
mpirun -np 96 --hostfile machinefile ./svm_struct_learn -c 100000000 -l 0 -w 1 --l 1 --w 10 --f 1 --p 1 --h 2 $DATADIR$CLS"_wrap.dat" $DATADIR$CLS"_unwrap.dat" $DATADIR$CLS"_neg.dat" $DATADIR$CLS".cad" | tee $DATADIR$CLS"_train.log"
