import datetime
import logging
import os
import pickle
from collections.abc import MutableMapping
from random import SystemRandom
from typing import Any, Dict, List, Optional

import numpy as np
from IPython import get_ipython

CLIP_DIV_PRED_MIN = 0.1
epsilon = 1e-6


def config_ipython():
    if get_ipython() is not None:
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
        get_ipython().run_line_magic("matplotlib", "inline")
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")


def config_logging(level=logging.DEBUG, name=None):
    logging.basicConfig()
    logging.getLogger(name).setLevel(level)


def timestamp() -> str:
    return str(datetime.datetime.now().isoformat())


def get_verbose_print_fct(verbose: bool):
    return print if verbose else lambda *a, **k: None


def create_random_str(length: Optional[int] = 32) -> str:
    letters = "abcdefghiklmnopqrstuvwwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    return "".join(SystemRandom().choices(letters, k=length))


def get_pretrained_model_base_path() -> str:
    return "/projects/controversial-stimuli/pretrained_models/ivans_rot_equivariant_spatialxfeatures"


def get_temp_dir() -> str:
    return os.environ.get("TEMP_DIR", ".")


def pairwise_combinations(lst):
    """Get pairwise combinations of elements in `lst`.

    Returns:
        NDArray: A 2D array of shape (len(lst), len(lst)) where each row is a pair of elements in `lst`.
    """
    pairwise_idc = np.array(np.meshgrid(lst, lst)).T.reshape(-1, 2)
    return pairwise_idc


def str_concat_from_dict(d: Dict[str, any]) -> str:
    return "___".join(["_".join([str(k), str(d[k])]) for k in sorted(d.keys())])


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def pickle_dump(path: str, name: str, obj: Any) -> None:
    with open(os.path.join(path, name), "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path: str, name: str) -> Any:
    with open(os.path.join(path, name), "rb") as f:
        return pickle.load(f)
