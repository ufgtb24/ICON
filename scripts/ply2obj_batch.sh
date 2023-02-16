#!/bin/bash
MODE=$1
# stringList=3dpeople,axyz,renderpeople,renderpeople_p27,humanalloy,thuman,thuman2
stringList=cape_raw
subject=00096
seq=shortlong_hips
# Use comma as separator and apply as pattern
for val in ${stringList//,/ }
do
    DATASET=$val
    echo "$DATASET START----------"
    # num_threads = 12
    # num_views = 36
    # resolution = 512
    # MODE = gen (process all subjects) | debug (only one subject)
    # PART = filename of render_list
    for val1 in ${subject//,/ }
    do
        SUBJECT=$val1

        for val2 in ${seq//,/ }
        do
            SEQ=$val2
            SAVE_DIR="./data/${DATASET}/${SUBJECT}/scans/$SEQ"
            mkdir -p $SAVE_DIR

            bash scripts/ply2obj.sh 4 $DATASET 36 512 $MODE $SUBJECT $SEQ
            echo "$DATASET END----------"
        done
    done
done