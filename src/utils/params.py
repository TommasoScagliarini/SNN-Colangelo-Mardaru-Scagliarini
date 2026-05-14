from typing import Dict, Union, Any
from pathlib import Path
from .global_variables import GlobalVariables
import yaml
from yaml.loader import UnsafeLoader
import skrl.resources.schedulers.torch as schedulers
import skrl.resources.preprocessors.torch as preprocessors
import torch.optim.lr_scheduler as torch_schedulers

from src.utils.scheduler import WarmStartCosineRestarts as _WarmStartCosineRestarts
from src.model.preprocessor import CrazyflieSNNPreprocessor as _SNNPrep

_CUSTOM_SCHEDULERS = {
    "WarmStartCosineRestarts": _WarmStartCosineRestarts,
}

_CUSTOM_PREPROCESSORS = {
    "CrazyflieSNNPreprocessor": _SNNPrep,
}


def simple_load_params(path: Union[Path, str], name: str) -> Dict[str, Any]:
    if not isinstance(path, Path):
        path = Path(path)
    with open(path / name, "r") as f:
        return yaml.load(f, Loader=UnsafeLoader)


def merge_params(params: Dict, default_params: Dict, env=None) -> Dict:
    """
    Turn YAML string identifiers into the matching skrl/torch objects.
      "KLAdaptiveLR"              -> skrl scheduler
      "LinearLR"                  -> torch scheduler
      "WarmStartCosineRestarts"   -> custom scheduler
      "RunningStandardScaler"     -> skrl preprocessor
      "CrazyflieSNNPreprocessor"  -> custom preprocessor
      size: null                  -> env.observation_space
      device: null                -> env.device
    """
    cfg = {**default_params}

    for k, v in params.items():

        if k == "learning_rate_scheduler":
            if v is None:
                cfg[k] = None
            else:
                sched = (
                    _CUSTOM_SCHEDULERS.get(v) or
                    getattr(schedulers, v, None) or
                    getattr(torch_schedulers, v, None)
                )
                cfg[k] = sched

        elif k in ["state_preprocessor", "value_preprocessor"]:
            cfg[k] = (
                _CUSTOM_PREPROCESSORS.get(v) or
                getattr(preprocessors, v, None)
            ) if v is not None else None

        elif isinstance(v, dict):
            tmp = {}
            for k_in, v_in in v.items():
                if k_in == "device" and env is not None:
                    tmp[k_in] = env.device
                elif v_in is None and k_in == "size":
                    tmp[k_in] = env.observation_space
                else:
                    tmp[k_in] = v_in
            cfg[k] = tmp

        else:
            cfg[k] = v

    return cfg


def save_params(params: Dict[str, Dict]) -> None:
    """Dump the parameters used by this run into the experiment directory."""
    gls = GlobalVariables()
    for name, content in params.items():
        with open(gls.path_global / (name + ".yaml"), "w") as f:
            yaml.dump(content, f)