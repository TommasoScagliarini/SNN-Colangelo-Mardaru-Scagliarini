import datetime
from pathlib import Path
from typing import Union
# import wandb


GLOBAL_PATH = Path("")
TASK = ""


def set_global_path(
        root_folder: Union[str, Path],
        path: Union[str, Path] = None,
        path_prefix: str = ""
    ) -> None:
    
    """
    Set GLOBAL_PATH to root_folder / path and create the directory tree.

    If path is None, it is generated as path_prefix + current datetime
    (format: YY-MM-DD_HH-MM-SS-us), ensuring a unique run directory per call.

    Args:
        root_folder: base directory for all run outputs.
        path:        explicit run subdirectory; auto-generated if None.
        path_prefix: prepended to the datetime string when path is None.
    """
    global GLOBAL_PATH
    if isinstance(root_folder, str):
        root_folder = Path(root_folder)
    if path is None:
        path = Path(path_prefix + datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
    GLOBAL_PATH = root_folder / path
    GLOBAL_PATH.mkdir(parents=True, exist_ok=True)


class GlobalVariables():
    def __init__(self):
        self.path_global: Path = GLOBAL_PATH