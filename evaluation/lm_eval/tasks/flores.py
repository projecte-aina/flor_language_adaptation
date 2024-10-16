
import collections
import datasets
import numpy as np

from lm_eval import metrics
from lm_eval.base import rf, Task

from lm_eval.metrics import mean
from lm_eval.tasks.dataset_paths import dataset_paths


class FloresBase(Task):
    VERSION = 1
    DATASET_PATH = dataset_paths["flores"] if "flores" in dataset_paths.keys() else "flores"
    DATASET_NAME = None

    cache = {}

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return [{"src": example["src"], "ref": example["ref"]} for example in self.dataset["dev"]]

    def test_docs(self):
        return [{"src": example["src"], "ref": example["ref"]} for example in self.dataset["devtest"]] 

    def doc_to_text(self, doc):
        code_to_language = {
            "cat_Latn": {
                "cat_Latn": "Català",
                "spa_Latn": "Catalán",
                "eng_Latn": "Catalan"
            },
            "spa_Latn": {
                "cat_Latn": "Espanyol",
                "spa_Latn": "Español",
                "eng_Latn": "Spanish"
            },
            "eng_Latn": {
                "cat_Latn": "Anglès",
                "spa_Latn": "Inglés",
                "eng_Latn": "English"
            }
        }

        source_code, target_code = self.DATASET_NAME.split("-")
        
        lower = lambda a: a.lower()
        first_upper = lambda a: a
        

        extra_initial_prompt_by_language = {
            "cat_Latn": "Tradueix la següent frase del català a l'",
            "spa_Latn": "Traduce la siguiente frase del español al ",
            "eng_Latn": "Translate this sentence from English to "
        }
        
        lower_by_language = {
            "cat_Latn": lower,
            "spa_Latn": lower,
            "eng_Latn": first_upper
        }

        source_language = code_to_language[source_code][source_code]
        target_language = code_to_language[target_code][source_code]
        
        source_text = doc["src"]

        return f"{extra_initial_prompt_by_language[source_code]}{lower_by_language[source_code](target_language)}:\n" + \
            f"{source_language}: {source_text}\n" + \
            f"{target_language}:"


    def should_decontaminate(self):
        return False  # TODO: to implement ...

    def doc_to_decontamination_query(self, doc):
        return doc["src"]

    def doc_to_target(self, doc):
        # This shows a single target, though there may be multiple targets in a lang test
        return " " + doc["ref"] # if isinstance(doc["ref"], str) else doc["ref"][0]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        return rf.greedy_until(ctx, {"until": ["\n"]})

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """

        # These metrics are corpus-level not sentence level, so we'll hide the
        # results in this dict and compute the corpus score in the aggregate method
        ref_pred = (doc["ref"], results)
        return {
            "bleu": ref_pred,
            "chrf": ref_pred,
            #"ter": ref_pred,
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "bleu": metrics.bleu,
            "chrf": metrics.chrf,
            #"ter": metrics.ter,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "bleu": True,
            "chrf": True,
            "ter": False,
        }

ca = "cat_Latn"
es = "spa_Latn"
en = "eng_Latn"

class flores_ca_es(FloresBase):
    DATASET_NAME=ca+"-"+es

class flores_es_ca(FloresBase):
    DATASET_NAME=es+"-"+ca

class flores_en_ca(FloresBase):
    DATASET_NAME=en+"-"+ca

class flores_ca_en(FloresBase):
    DATASET_NAME=ca+"-"+en

class flores_es_en(FloresBase):
    DATASET_NAME=es+"-"+en

class flores_en_es(FloresBase):
    DATASET_NAME=en+"-"+es

