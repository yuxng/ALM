#!/bin/bash

CLS=car
DATADIR=../Main/data/

# Perform tasks
echo "mpirun -np 96 --hostfile machinefile ./svm_struct_classify $DATADIR"val.tst" $DATADIR$CLS".cad" $DATADIR$CLS"_final.mod" $DATADIR"val.pre""

mpirun -np 96 --hostfile machinefile ./svm_struct_classify $DATADIR"val.tst" $DATADIR$CLS".cad" $DATADIR$CLS"_final.mod" $DATADIR"val.pre"
