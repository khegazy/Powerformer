#!/bin/bash

for MASK in zeta step
do
    for SCRIPT in etth1 etth2 ettm1 ettm2 illness
    do
        echo ${SCRIPT}
        sh scripts/PatchTST/sbatch_submit.sh ${SCRIPT} ${MASK}
    done
done