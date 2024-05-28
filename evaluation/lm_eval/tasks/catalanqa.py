"""
CatananQA
"""
import datasets
from lm_eval.tasks.squad import XQuAD
from math import exp
from lm_eval.base import rf, Task
from functools import partial
from packaging import version
from lm_eval.tasks.dataset_paths import dataset_paths
from lm_eval.extra_metrics import squad, squad_v2


_CITATION = """
"""


def _squad_metric(predictions, references):
    squad_metric = datasets.load_metric(squad_v2)
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)

    return _squad_metric(predictions=predictions, references=references).get(key, 0)


class CatalanQA(XQuAD):
    DATASET_PATH = dataset_paths["catalanqa"] if "catalanqa" in dataset_paths.keys() else "projecte-aina/catalanqa"
    DATASET_NAME = "catalanqa"

    def doc_to_target(self, doc):
        answer = doc["answers"][0]["text"]
        if len(answer) == 0:
            answer = "unanswerable"
            print("unanswerable")
        return " " + answer

    def doc_to_text(self, doc):
        return (
            #"Title: " # No title in XQuAD
            #+ doc["title"]
            #+ "\n\n"
            "Context: "
            + doc["context"]
            + "\n\n"
            + "Pregunta: "
            + doc["question"]
            + "\n\n"
            + "Resposta:"
        )
