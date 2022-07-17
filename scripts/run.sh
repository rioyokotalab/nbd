#!/bin/bash
set -x
export OMP_NUM_THREADS=1
BIN=~/nbd/build/lorasp
LOGS=~/nbd/log
SCRIPT=~/nbd/scripts
set +x

rm -rf $LOGS && mkdir $LOGS
echo Running serial tests..
set -x
mpirun -n 1 $BIN 2048 2 256 1.e-10 100 2000 >> $LOGS/serial.log
mpirun -n 1 $BIN 4096 2 256 1.e-10 100 2000 >> $LOGS/serial.log
mpirun -n 1 $BIN 8192 2 256 1.e-10 100 2000 >> $LOGS/serial.log
mpirun -n 1 $BIN 16384 2 256 1.e-10 100 2000 >> $LOGS/serial.log
mpirun -n 1 $BIN 32768 2 256 1.e-10 100 2000 >> $LOGS/serial.log
set +x
echo Finish!

echo Running strong scaling tests..
set -x
mpirun -n 1 $BIN 16384 2 256 1.e-10 100 2000 >> $LOGS/strong.log
mpirun -n 2 $BIN 16384 2 256 1.e-10 100 2000 >> $LOGS/strong.log
mpirun -n 4 $BIN 16384 2 256 1.e-10 100 2000 >> $LOGS/strong.log
mpirun -n 8 $BIN 16384 2 256 1.e-10 100 2000 >> $LOGS/strong.log
mpirun -n 16 $BIN 16384 2 256 1.e-10 100 2000 >> $LOGS/strong.log
set +x
echo Finish!

echo Processing logs file..
gcc $SCRIPT/logs2csv.c -o $LOGS/logs2csv
$LOGS/logs2csv $LOGS/serial.log $LOGS/serial.csv
$LOGS/logs2csv $LOGS/strong.log $LOGS/strong.csv
echo Finish!

echo Generating plots..
python3 $SCRIPT/plot_serial.py $LOGS/serial.csv $LOGS/factor_serial.png
python3 $SCRIPT/plot_strong.py $LOGS/strong.csv $LOGS/factor_strong.png
python3 $SCRIPT/plot_memory.py $LOGS/serial.csv $LOGS/mem_serial.png
python3 $SCRIPT/plot_memory.py $LOGS/strong.csv $LOGS/mem_strong.png
python3 $SCRIPT/plot_comm.py $LOGS/strong.csv $LOGS/comm_strong.png

echo Finish!
