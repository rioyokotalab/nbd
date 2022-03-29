#!/bin/bash
#PJM -L "node=16x16"
#PJM -L "rscgrp=small"
#PJM -L "elapse=5:00:00"
#PJM --llio cn-cache-size=1Gi
#PJM --llio sio-read-cache=on
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -x PJM_LLIO_GFSCACHE=/vol0003
#PJM --mpi "proc=1024"
#PJM --mpi "max-proc-per-node=4"
#PJM --mpi  "rank-map-bynode"
#PJM -s

source /vol0004/apps/oss/spack/share/spack/setup-env.sh
export MODULEPATH=/vol0004/apps/arm-lib/modulefiles:$MODULEPATH
export MODULEPATH=/vol0004/apps/oss/spack/share/spack/modules/linux-rhel8-a64fx:$MODULEPATH

spack load /vgsk3aw
make

procs=1024

for N in 131072 262144 524288 1048576 2097152; do
	for nleaf 256; do
		mpiexec -std out/output_$procs.txt -n $procs ./lorasp $N $nleaf 1 3
	done
done