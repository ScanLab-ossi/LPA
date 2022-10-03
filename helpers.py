from __future__ import annotations

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from corpora import Matrix


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.now()
        result = f(*args, **kw)
        te = datetime.now()
        print(f"func: {f.__name__} took: {te-ts}")
        return result

    return wrap


def write(path: Path, *args: Tuple[pd.DataFrame | Matrix, str]):
    for table, name in args:
        if isinstance(table, pd.DataFrame):
            table.to_csv(path / f"{name}.csv", index=False)
        else:
            with open(path / f"{name}.npy", "wb") as f:
                np.save(f, table.matrix)
        print(f"wrote {name}")


def read(path: Path, name_with_ext: str) -> pd.DataFrame | np.array:
    if name_with_ext[-3:] == "npy":
        return np.load(path / name_with_ext)
    else:
        return pd.read_csv(
            path / name_with_ext,
            parse_dates=(["date"] if name_with_ext != "dvr.csv" else []),
        )
