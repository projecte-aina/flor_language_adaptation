from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from lm_eval import utils
from lm_eval.tasks.pawsx import PAWSX_ca
from lm_eval.tasks.dataset_paths import dataset_paths

class Parafraseja(PAWSX_ca):
    VERSION = 0
    DATASET_PATH = dataset_paths["parafraseja"] if "parafraseja" in dataset_paths.keys() else "projecte-aina/Parafraseja"
    DATASET_NAME = None

