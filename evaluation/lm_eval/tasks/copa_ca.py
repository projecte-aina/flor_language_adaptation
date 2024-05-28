"""
XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning
https://ducdauge.github.io/files/xcopa.pdf

The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across languages.
The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around the globe.
The dataset is challenging as it requires both the command of world knowledge and the ability to generalise to new languages.
All the details about the creation of XCOPA and the implementation of the baselines are available in the paper.

Homepage: https://github.com/cambridgeltl/xcopa
"""


_CITATION = """

Professional translation of COPA.
@inproceedings{ponti2020xcopa,
  title={{XCOPA: A} Multilingual Dataset for Causal Commonsense Reasoning},
  author={Edoardo M. Ponti, Goran Glava\v{s}, Olga Majewska, Qianchu Liu, Ivan Vuli\'{c} and Anna Korhonen},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020},
  url={https://ducdauge.github.io/files/xcopa.pdf}
}
"""
import numpy as np
import sklearn
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, acc_all, metric_max_over_ground_truths, yesno
from lm_eval.utils import general_detokenize
from lm_eval.tasks.dataset_paths import dataset_paths


class CopaCa(Task):
    VERSION = 0
    DATASET_PATH = dataset_paths["copa_ca"] if "copa_ca" in dataset_paths.keys() else "projecte-aina/COPA-ca"
    DATASET_NAME = "copa-ca"
    CAUSE = "perquè" #"because"
    EFFECT = "i per tant" #"therefore"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        # Drop the period
        connector = {
            "cause": self.CAUSE,
            "effect": self.EFFECT,
        }[doc["question"]]
        return doc["premise"].strip()[:-1] + f" {connector}"

    def doc_to_target(self, doc):
        correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
        # Connect the sentences
        return " " + self.convert_choice(correct_choice)

    def construct_requests(self, doc, ctx):
        choice1 = " " + self.convert_choice(doc["choice1"])
        choice2 = " " + self.convert_choice(doc["choice2"])

        ll_choice1, _ = rf.loglikelihood(ctx, choice1)
        ll_choice2, _ = rf.loglikelihood(ctx, choice2)

        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1.0 if pred == gold else 0.0

        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]

