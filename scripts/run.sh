#!/bin/bash

export OMP_NUM_THREADS=1
BIN=~/nbd/build/lorasp
LOGS=~/nbd/log

rm -rf $LOGS && mkdir $LOGS
echo Running serial tests..
mpirun -n 1 $BIN 2048 2 >> $LOGS/serial.log
mpirun -n 1 $BIN 4096 2 >> $LOGS/serial.log
mpirun -n 1 $BIN 8192 2 >> $LOGS/serial.log
mpirun -n 1 $BIN 16384 2 >> $LOGS/serial.log
mpirun -n 1 $BIN 32768 2 >> $LOGS/serial.log
echo Finish!

echo Running strong scaling tests..
mpirun -n 1 $BIN 16384 2 >> $LOGS/strong.log
mpirun -n 2 $BIN 16384 2 >> $LOGS/strong.log
mpirun -n 4 $BIN 16384 2 >> $LOGS/strong.log
mpirun -n 8 $BIN 16384 2 >> $LOGS/strong.log
mpirun -n 16 $BIN 16384 2 >> $LOGS/strong.log
echo Finish!

echo Processing logs file..
gcc logs2csv.c -o $LOGS/logs2csv
$LOGS/logs2csv $LOGS/serial.log $LOGS/serial.csv
$LOGS/logs2csv $LOGS/strong.log $LOGS/strong.csv
echo Finish!

echo Generating plots..
python3 plot_serial.py $LOGS/serial.csv $LOGS/factor_serial.png
python3 plot_strong.py $LOGS/strong.csv $LOGS/factor_strong.png
echo Finish!