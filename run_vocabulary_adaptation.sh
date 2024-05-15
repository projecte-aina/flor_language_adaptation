#!/bin/bash

echo "[BASH] Setting up virtual environment ..."
source use_env.sh

echo "[BASH] Setting up cache folder ..."
export HF_DATASETS_CACHE="./cache"

BIG_SOURCE_MODEL_DIRECTORY="./models/bloom-1b3/"  # TODO: document the need of downloading model
TARGET_TOKENIZER="./tokenizers/FLOR-1.3B/"  # TODO: document the need of downloading the tokenizer
OUTPUT_DIRECTORY="./models/"
NAME="FLOR-1.3B"
STRATEGY="matching"
PAD_TOKEN="<|endoftext|>"

echo "[BASH] Running ..."
python vocabulary_adaptation.py \
    --big_source_model_directory $BIG_SOURCE_MODEL_DIRECTORY \
    --target_tokenizer $TARGET_TOKENIZER \
    --output_directory $OUTPUT_DIRECTORY \
    --name $NAME \
    --strategy $STRATEGY \
    --pad_token $PAD_TOKEN \
    --save_embeddings

