#!/bin/bash
NUM_THREADS=$1
DATASET=$2
NUM_VIEWS=$3
SIZE=$4
MODE=$5
SUBJECT=$6
SEQ=$7

PYTHON_SCRIPT="scripts/p2o2.py"

if [[ $MODE == "gen" ]]; then
    echo "processing all the subjects"
    # render all the subjects
    cat ./data/$DATASET/$SUBJECT/$SEQ.txt | shuf | xargs -P$NUM_THREADS -I {} python $PYTHON_SCRIPT -s $SUBJECT -q $SEQ -f {}
fi

if [[ $MODE == "debug" ]]; then
    echo "Debug renderer"
    # render only one subject
    SAVE_DIR="./debug/${DATASET}_${NUM_VIEWS}views"
    mkdir -p $SAVE_DIR

    if [[ $DATASET == "thuman2" ]]; then
        SUBJECT="0001"
        echo "Rendering $DATASET $SUBJECT"
    fi

    python $PYTHON_SCRIPT -s $SUBJECT -o $SAVE_DIR -r $NUM_VIEWS -w $SIZE
fi
