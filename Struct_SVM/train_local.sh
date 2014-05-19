#!/bin/bash

CLS=car
DATADIR=../Main/data_debug/

# Perform tasks
mpirun -np 12 ./svm_struct_learn -c 100000000 -l 0 -w 1 --l 100 --w 10 --f 0 --p 2 --h 0 $DATADIR$CLS"_wrap.dat" $DATADIR$CLS"_unwrap.dat" $DATADIR$CLS"_neg.dat" $DATADIR$CLS".cad" | tee $DATADIR$CLS"_train.log"
