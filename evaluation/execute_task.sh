#!/bin/bash                                                                                                                                                                                                      

export HF_HOME="./cache"

source venv/bin/activate

model=$1
dataset=$2
few_shot=$3

python main.py  \
	--model hf-causal \
	--model_args pretrained=$model,trust_remote_code=True \
	--tasks $dataset \
	--num_fewshot $few_shot \
	--output_path results/$(basename ${model})/results:$(basename ${model}):${dataset}:${few_shot}-shot.txt \
	--no_cache


