#!/bin/bash

#for SCRIPT in etth1 
for SCRIPT in etth1 etth2 ettm1 ettm2 illness
do
    echo ${SCRIPT}
    sh scripts/PatchTST/sbatch_submit.sh ${SCRIPT}
done