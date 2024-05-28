import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from lm_eval import utils
from . import xnli_v2
from lm_eval.tasks.dataset_paths import dataset_paths

class Teca(xnli_v2.XNLIBase):
    DATASET_PATH = dataset_paths["teca"] if "teca" in dataset_paths.keys() else "projecte-aina/teca"
    DATASET_NAME = "teca"

    QUESTION_WORD = "correcte"
    ENTAILMENT_LABEL = "Sí"
    NEUTRAL_LABEL = "A més"
    CONTRADICTION_LABEL = "No"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["dev"]

    def test_docs(self):
        return self.dataset["test"]


