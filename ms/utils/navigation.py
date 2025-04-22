import json
import os
from abc import ABC, abstractmethod
from os.path import join
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure


def get_project_path() -> str:
    return str(Path(__file__).parent.parent.parent)

def get_config_path() -> str:
    return pjoin(get_project_path(), "config")

def get_prefix(s: str) -> str:
    return s.split("__")[0]

def get_suffix(s: str) -> str:
    return s.split("__")[-1]

def has_suffix(s: str) -> bool:
    return "__" in s

def pjoin(*args) -> Path:
    return Path(join(*args))

def get_file_name(prefix: str, suffix: str | None = None):
    if suffix is not None:
        res = f"{prefix}__{suffix}"
    else:
        res = f"{prefix}"
    return res


class ResultType(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def save(self, path: Path, data: Any) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> Any:
        pass


class JSONType(ResultType):
    name = "json"

    def save(self, path: Path, data: dict) -> None:
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path) -> dict:
        with open(path, "r") as f:
            data = json.load(f)
        return data

class CSVType(ResultType):
    name = "csv"

    def save(self, path: Path, data: pd.DataFrame) -> None:
        data.to_csv(
            path_or_buf=path,
            header=True,
        )

    def load(self, path: Path) -> pd.DataFrame:
        data = pd.read_csv(path, index_col=False)
        return data

class PNGType(ResultType):
    name = "png"

    def save(self, path: Path, data: Figure) -> None:
        data.savefig(path)

    def load(self, path: Path) -> Figure:
        return Figure()

result_types = {
    "json": JSONType(),
    "csv": CSVType(),
    "png": PNGType(),
}

def load(
        path: Path,
        file_type: str | None = None
) -> Any:
    if file_type is None:
        file_type = path.name.split(".")[-1]
    return result_types[file_type].load(path=path)

def save(
        data: Any,
        path: Path,
        file_type: str | None = None,
) -> None:
    if file_type is None:
        file_type = path.name.split(".")[-1]
    path.parent.mkdir(exist_ok=True, parents=True)
    result_types[file_type].save(path=path, data=data)

def get_path(
        folders: list[str],
        file_name: str,
        file_type: str,
) -> Path:
    path = pjoin(
        get_project_path(),
        *folders,
        f"{file_name}.{file_type}",
    )
    return Path(path)

def rewrite_decorator(func):
    def wrapper(*args, **kwargs):
        save_path = kwargs.get("save_path", Path())
        to_rewrite = kwargs.get("to_rewrite", False)
        save_idx = kwargs.get("save_idx", 0)

        if not to_rewrite and save_path.exists():
            print(f"File {save_path} already exists. Skipping...")
            return load(save_path)
        res = func(
            *args,
            **kwargs
        )
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if isinstance(res, (pd.DataFrame, dict)):
                save(data=res, path=save_path)
            else:
                save(data=res[save_idx], path=save_path)
        return res
    return wrapper
