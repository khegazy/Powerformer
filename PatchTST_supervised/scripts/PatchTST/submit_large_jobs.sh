#!/bin/bash

for SCALE in 0.5 1 2 5 10
do
    #for SCRIPT in electricity traffic weather
    for SCRIPT in weather
    do
        echo ${SCRIPT}
        sh scripts/PatchTST/sbatch_submit.sh ${SCRIPT} ${SCALE}
    done
done