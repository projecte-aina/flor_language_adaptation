#!/usr/bin/env bash

source use_env.sh

WORKING_PATH=$1
LANGUAGE=$2
INPUT_FILE=$WORKING_PATH/output.txt
INPUT_ONION=$WORKING_PATH/output.vert
OUTPUT_ONION=$WORKING_PATH/output.onion
OUTPUT_FILE=$WORKING_PATH/output.clean

python clean.py --language $LANGUAGE --input_path "$INPUT_FILE" --input_format "default" --output_path "$INPUT_ONION" --output_format "onion"
onion -sm -n 5 -p doc -t 0.5 "$INPUT_ONION"  > "$OUTPUT_ONION"
onion -sm -n 5 -t 0.8 "$OUTPUT_ONION"  > "$OUTPUT_ONION"
python clean.py --language $LANGUAGE --input_path "$OUTPUT_ONION" --input_format "onion" --output_path "$OUTPUT_FILE" --output_format "default"


