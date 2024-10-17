#!/bin/bash

for MASK in zeta 
do
    for SCALE in 0.1 0.5 1 2 5 10
    do
        #for SCRIPT in electricity traffic weather
        for SCRIPT in weather
        do
            echo ${SCRIPT}
            sh scripts/PatchTST/sbatch_submit.sh ${SCRIPT} ${MASK} ${SCALE}
        done
    done
done