"""
professional translation into Catalan of Winograd NLI dataset as published in GLUE Benchmark.
The Winograd NLI dataset presents 855 sentence pairs,
in which the first sentence contains an ambiguity and the second one a possible interpretation of it.
The label indicates if the interpretation is correct (1) or not (0).
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
import platform
from lm_eval.tasks.xnli_v2 import XNLIBase
from lm_eval.tasks.glue import WNLI
from lm_eval.tasks.dataset_paths import dataset_paths


_CITATIONS = """

"""


class WNLIBase(XNLIBase):
    VERSION = 0
    DATASET_NAME = None
    _DATASET_NAME = None
    DATASET_PATH = None

    QUESTION_WORD = None  # 'right'
    ENTAILMENT_LABEL = None  # 'Yes'
    NEUTRAL_LABEL = None  # 'Also'
    CONTRADICTION_LABEL = None  # 'No'
    TEMPLATE = None

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):

        self.DATASET_NAME = None

        super().__init__(data_dir, cache_dir, download_mode)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_target(self, doc):
        # True = entailment
        # False = not_entailment
        return " {}".format({0: self.CONTRADICTION_LABEL, 1: self.ENTAILMENT_LABEL}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true = rf.loglikelihood_rolling(ctx.replace("[MASK]", self.ENTAILMENT_LABEL))
        ll_false = rf.loglikelihood_rolling(
            ctx.replace("[MASK]", self.CONTRADICTION_LABEL)
        )

        return ll_true, ll_false

    def process_results(self, doc, results):
        ll_true, ll_false = results
        pred = ll_false > ll_true
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}



    def doc_to_text(self, doc):
        # Example:
        # The girl that can help me is all the way across town, right? Yes, The girl I need help from lives a ways away.
        # [MASK] is replaced with ENTAILMENT_LABEL, NEUTRAL_LABEL, or CONTRADICTION_LABEL
        return (
            doc["sentence1"].strip()[:-1]
            + ", "
            + self.QUESTION_WORD
            + "? [MASK], "
            + doc["sentence2"][0].lower() + doc["sentence2"][1:]
        )


class WNLI_es(WNLI):
    DATASET_NAME = "es"
    DATASET_PATH = dataset_paths["wnli_es"] if "wnli_es" in dataset_paths.keys() else "PlanTL-GOB-ES/wnli-es"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        super().__init__(data_dir, cache_dir, download_mode)

    def doc_to_text(self, doc):
        return "{}\nPregunta: {} ¿Verdadero o falso?\nRespuesta:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = not_entailment
        return " {}".format({0: "Falso", 1: "Verdadero"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " Verdadero")
        ll_false, _ = rf.loglikelihood(ctx, " Falso")
        return ll_true, ll_false

class WNLI_ca(WNLI):  # Catalan
    DATASET_NAME = "ca"
    DATASET_PATH = dataset_paths["wnli_ca"] if "wnli_ca" in dataset_paths.keys() else "projecte-aina/wnli-ca"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        super().__init__(data_dir, cache_dir, download_mode)

    def doc_to_text(self, doc):
        return "{}\nPregunta: {} Cert o fals?\nResposta:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = not_entailment
        return " {}".format({0: "Fals", 1: "Cert"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " Cert")
        ll_false, _ = rf.loglikelihood(ctx, " Fals")
        return ll_true, ll_false

