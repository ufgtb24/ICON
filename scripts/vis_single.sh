#!/bin/bash
NUM_THREADS=$1
DATASET=$2
NUM_VIEWS=$3
MODE=$4
SUBJECT=$5
SEQ=$6

PYTHON_SCRIPT="scripts/vis_single.py"

if [[ $MODE == "gen" ]]; then
    echo "processing all the subjects"
    # compute visibility for all the subjects
    LOG_FILE="./log/vis/${DATASET}-${NUM_VIEWS}-${PART}.txt"
    SAVE_DIR="./data/${DATASET}/${SUBJECT}_${NUM_VIEWS}views/$SEQ"
    mkdir -p $SAVE_DIR
    mkdir -p "./log/vis/"
    cat ./data/$DATASET/$SUBJECT/$SEQ.txt | shuf | xargs -P$NUM_THREADS -I {} python $PYTHON_SCRIPT  -s $SUBJECT -q $SEQ -f {} -o $SAVE_DIR -r $NUM_VIEWS -m $MODE> $LOG_FILE
fi

if [[ $MODE == "debug" ]]; then
    echo "Debug visibility"
    # compute visibility for only one subject
    SAVE_DIR="./debug/${DATASET}_${NUM_VIEWS}views"
    mkdir -p $SAVE_DIR

    if [[ $DATASET == "thuman2" ]]; then
        SUBJECT="0001"
        echo "Visibility $DATASET $SUBJECT"
    fi

    python $PYTHON_SCRIPT -s $SUBJECT -o $SAVE_DIR -r $NUM_VIEWS -m $MODE
fi
