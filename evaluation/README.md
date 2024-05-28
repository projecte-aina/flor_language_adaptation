# Evaluation
This folder contains the scripts and data used to evaluate the models presented in the paper.

## Framework
We used the lm-evaluation-harness framework from EleutherAI, last updated on 17/06/2023. The scripts in /lm-eval and main.py are taken directly from it.

## Tasks and data
The evaluation was carried out on several Catalan, Spanish and English tasks covering the following task categories: Reading Comprehension, Question Answering (QA), Reasoning, NLI, Paraphrase Identification and Translation. All the datasets used can be found online at the link provided and in the /data folder (zipped) together with the necessary dataloaders ( included in this repository for reproducibility purposes). Each dataset has its own licence, which has been copied to the table from the original resource.

The following table shows the task names, categories and some comments about the task or its implementation:

| Task Category | Task Name | Comment | License |
|:-------------:|:---------:|:-------:|:-------:|
| Reading Comprehension | [belebele_en, belebele_ca, belebele_es](https://huggingface.co/datasets/facebook/belebele) | The task implementation differs from the current official one, as there was no implementation on the lm-evaluation harness at the time of this work. | CC-BY-SA-4.0
| QA            | [xquad_en](https://github.com/google-deepmind/xquad), [xquad_ca](https://huggingface.co/datasets/projecte-aina/xquad-ca), [xquad_es](https://github.com/google-deepmind/xquad) | Extractive QA. Implementation based on SQuAD's. | CC-BY-SA-4.0
| QA            | [catalanqa](https://huggingface.co/datasets/projecte-aina/catalanqa) | Extractive QA; Catalan is the original language. Task implementation as xquad_ca. | CC-BY-SA-4.0
| QA            | [coqcat](https://huggingface.co/datasets/projecte-aina/CoQCat)    | Conversational QA; Catalan is the original language. Task implementation based on COQA's (en). | CC-BY-NC-ND-4.0
| NLI           | [xnli_v2_en](https://huggingface.co/datasets/xnli), [xnli_v2_ca](https://huggingface.co/datasets/projecte-aina/xnli-ca), [xnli_v2_es](https://huggingface.co/datasets/xnli)    | The task implementaiton is based on the official one, with some added text preprocessing.  | CC-BY-NC-4.0
| NLI           | [teca](https://huggingface.co/datasets/projecte-aina/teca)    | Catalan is the original language. Task implementation as xnli_v2_ca | CC-BY-NC-ND-4.0
| Paraphrase Identification           | [paws_en](https://huggingface.co/datasets/paws), [paws_ca](https://huggingface.co/datasets/projecte-aina/PAWS-ca)    | English version already implemented in lm-evaluation-harness. | Other - consult [here](https://huggingface.co/datasets/paws#licensing-information).
| Paraphrase Identification           | [parafraseja](https://huggingface.co/datasets/projecte-aina/Parafraseja)    | Catalan is the original language. Task implementation as paws_ca. | CC-BY-NC-ND-4.0
| Commonsense Reasoning           | [xstorycloze_en](https://huggingface.co/datasets/juletxara/xstory_cloze), [xstorycloze_ca](https://huggingface.co/datasets/projecte-aina/xstorycloze_ca)    |  English version already implemented in lm-evaluation-harness. | CC-BY-4.0
| Commonsense Reasoning           | [copa_ca](https://huggingface.co/datasets/projecte-aina/COPA-ca), [copa_en](https://huggingface.co/datasets/aps/super_glue)    |  English version already implemented in lm-evaluation-harness. | CC-BY-SA-4.0
| Translation           | [flores_ca_en, flores_ca_es, flores_en_ca, flores_es_ca, flores_es_en, flores_en_es](https://huggingface.co/datasets/facebook/flores)    |  Implemented from scratch. | CC-BY-SA-4.0


## Steps to run the evaluation
1. Use `pip install -r requirements.txt` to install the required libraries.
2. Unzip `offline_data`.
3. Create a `results` and a `cache` folder.
4. For each task you want to evaluate, do `bash execute_task.sh <model> <task> <num_fewshot>`, specifying the model path, task name, and the number of few-shot examples (we used 5-shot for our evaluations).  If you want to send all tasks using sbatch, run the `launch_tasks.sh` script after adding the required sbatch header to `execute_tasks.sh`.
5. When the evaluation is finished and the results are stored in the `results` folder, run `extract_results.py` to create an Excel sheet summarising the results.
