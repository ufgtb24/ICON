#!/bin/bash
NUM_THREADS=$1
DATASET=$2
NUM_VIEWS=$3
SIZE=$4
MODE=$5
SUBJECT=$6
SEQ=$7

PYTHON_SCRIPT="scripts/render_single_cape.py"

if [[ $MODE == "gen" ]]; then
    echo "processing all the subjects"
    # render all the subjects
    LOG_FILE="./log/render/${DATASET}-${NUM_VIEWS}-${SIZE}-${PART}.txt"
    SAVE_DIR="./data/${DATASET}/${SUBJECT}_${NUM_VIEWS}views/$SEQ"
    mkdir -p $SAVE_DIR
    mkdir -p "./log/render/"
    cat ./data/$DATASET/$SUBJECT/$SEQ.txt | shuf | xargs -P$NUM_THREADS -I {} python $PYTHON_SCRIPT -s $SUBJECT -q $SEQ -f {} -o $SAVE_DIR -r $NUM_VIEWS -w $SIZE
#    PYTHONUNBUFFERED=1 python -u $PYTHON_SCRIPT -s $SUBJECT -q $SEQ -f 'shortlong_hips.000001.ply' -o $SAVE_DIR -r $NUM_VIEWS -w $SIZE
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
