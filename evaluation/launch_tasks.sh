#!/bin/bash

models=(
	projecte-aina/FLOR-760M
	projecte-aina/FLOR-1.3B
)
datasets=( belebele_ca belebele_es belebele_en flores_ca_es flores_ca_en flores_es_ca flores_es_en flores_en_ca flores_en_es xnli_v2_ca xnli_v2_es xnli_v2_en xquad_ca xquad_es xquad_en parafraseja teca copa_ca coqcat copa_en catalanqa pawsx_ca pawsx_en pawsx_es xstory_cloze_en xstory_cloze_es )

num_fewshots=( 5 )

for model in ${models[@]}
do
    for task in ${datasets[@]}
    do
        for num_fewshot in ${num_fewshots[@]}
        do
            sbatch execute_task.sh $model $task $num_fewshot
        done
    done
done
