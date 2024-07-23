# FLOR, Language Adaptation

This is the code used to prepare the data, train and evaluate the models in the paper [FLOR: On the Effectiveness of Language Adaptation](https://aclanthology.org/2024.lrec-main.650/). 

## Install

You need to:
- TODO: python requirements.
- Install [Onion](https://corpus.tools/wiki/Onion)
- Download `lid.176.bin`: `wget -P data_preprocessing https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`
- Download `nltk.punkt`. TODO: possibly make a script 

## Data Processing
- `data_processing/run_clean.py`: example of pipeline to do the full data processing.  # TODO: explain clean-onion-clean sandwich.
- `data_processing/parser.py`: utility to parse input parameters.
- `clean.py`: this script processes documents and filters them on 3 levels:
    - `document`: Available filters:
        - Minimum number of sentences (set `min_sentences_per_document`)
        - Maximum number of ellipsis (set `max_number_ellipsis_per_document`)
        - Language (set `language`)
    - `paragraph`: Available filter:
        - Minumum words number (set `min_number_of_words_per_paragraph`)
    - `sentence`: Available filter:
        - First char is not lowercase
        - Last character is allowed (set `allowed_end_of_sentence`)
    TODO: again, explain sandwich (document-paragraph-sentence-paragraph-document).
- `document.py`: defines Document, Paragraph and Sentence classes, necessary for `clean.py`.
- `input_formats.py` and `output_formats.py`: input and output format functions required to run the pipeline.
TODO: add an example on how to run this.
TODO: I created a dummy, but it needs to be cleaned, it does use filters for example.


## Vocabulary adaptation
 
In order to perform vocabulary adaptation of a model from a source tokenizer to a new target one, you can use `vocabulary_adaptation.py`.
You can choose the vocabulary adaptation strategy with the argument `--strategy`. The options are:
- `matching (default)` ([Transfer Learning in Multilingual Neural Machine Translation with Dynamic Vocabulary](https://aclanthology.org/2018.iwslt-1.8/))
- `improved` ([Efficient Language Model Training through Cross-Lingual and Progressive Transfer Learning](https://arxiv.org/abs/2301.09626))
- `lstsq` ([As Good as New. How to Successfully Recycle English GPT-2 to Make Models for Other Languages](https://aclanthology.org/2021.findings-acl.74/))
- `orthogonal_procrustes` ([As Good as New. How to Successfully Recycle English GPT-2 to Make Models for Other Languages](https://aclanthology.org/2021.findings-acl.74/))
- `knn (NOT IMPLEMENTED)` ([As Good as New. How to Successfully Recycle English GPT-2 to Make Models for Other Languages](https://aclanthology.org/2021.findings-acl.74/))
- `wechsel (NOT IMPLEMENTED)` ([WECHSEL: Effective initialization of subword embeddings for
cross-lingual transfer of monolingual language models](https://aclanthology.org/2022.naacl-main.293.pdf))
 
Other arguments for this script are:
- `--small_source_model_directory`: Main folder of the small source model. Only required if strategy is `lstsq` or `orthogonal_procrustes`.
- `--big_source_model_directory`: Main folder of the small source model. 
- `--small_target_model_directory`: Main folder of the small target model. Only required if strategy is `improved`, `lstsq` or `orthogonal_procrustes`.
- `--source_tokenizer`: (OPTIONAL) Main folder of the source tokenizer. Default is the same folder as big source model. 
- `--target_tokenizer`: (OPTIONAL) Main folder of the target tokenizer. Default is the same folder as small target model (if given).
- `--output_directory`: (OPTIONAL) Path to the output directory. Default is `./new_models`.
- `--name`: (OPTIONAL) Name of the new model. Default is `new_{base_model_name}`.
- `--pad_token`: Allows to set the PAD token of the new tokenizer and match model config accordingly.
- `--eos_token`: Allows to set the EOS token of the new tokenizer and match model config accordingly.
- `--bos_token`: Allows to set the BOS token of the new tokenizer and match model config accordingly.
- `--debug`: (OPTIONAL) Faster, full models are not loaded, only embeddings (if possible). Results are not saved, but a console is opened at the end of run for further testing. NOT RECOMMENDED for anything else then debugging. Default is False.
- `--force_overwrite`: (OPTIONAL) Force overwrite if output folder already exists. Recommended if run on background. 
- `--save_embeddings`: (OPTIONAL) Save embeddings of all models for future use.

In the naming of the first three parameters, `big` refers to target model size, while `small` is the size of the support models. Notice that for some strategies, small models are optional.
 
An example on how to run this script is `run_vocabulary_adaptation.sh` used to adapt BLOOM-7.1B model to a Catalan-Spanish tokenizer.


## Evaluation
TODO

