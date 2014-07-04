#!/bin/bash

CLS=car
DATADIR=../Main/data_aspect/

# Perform tasks
cat test_mpi.sh
mpirun -np 96 --hostfile machinefile ./svm_struct_classify $DATADIR"val.tst" $DATADIR$CLS".cad" $DATADIR$CLS"_final.mod" $DATADIR"val.pre"
