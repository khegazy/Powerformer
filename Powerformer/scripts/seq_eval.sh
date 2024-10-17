#!/bin/bash

for SCRIPT in etth1 etth2 ettm1 ettm2 weather
do
    echo ${SCRIPT}
    sh scripts/PatchTST/${SCRIPT}.sh
    for MASK in zeta step
    do
        sh scripts/PatchTST/${SCRIPT}.sh --attn_decay_type ${MASK}
    done
done