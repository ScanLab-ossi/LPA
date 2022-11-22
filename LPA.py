from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from scipy.special import lambertw
from scipy.spatial.distance import cdist
import bottleneck as bn

from algo import entropy, KLD_distance_overused
from helpers import read, write, timing


class Matrix:
    def __init__(self, matrix: np.array):
        self.matrix = matrix
        self.normalized = False

    def __bool__(self):
        return True if hasattr(self, "matrix") else False

    def _get_epsilon(self, lambda_=1) -> float:
        """
        λ is the contibution to the entropy by the terms with probability ε
        ε ≈ six orders of magnitude smaller than λ
        """
        m = np.count_nonzero((self.matrix == 0), axis=1).max()
        if lambda_ > m / (np.e * np.log(2)) or lambda_ <= 0:
            raise ValueError
        s = entropy(self.matrix).sum(axis=1).max()
        res = np.minimum(
            np.e ** lambertw(-lambda_ * np.log(2) / m, k=-1).real, 2 ** (-s)
        )
        return res

    def epsilon_modification(
        self,
        epsilon: float | None = None,
        lambda_: float | int = 1,
        threshold: float = 0,
    ):
        if not epsilon:
            epsilon = self._get_epsilon(lambda_)
        beta = 1 - epsilon * np.count_nonzero(self.matrix <= threshold, axis=1)
        self.matrix = self.matrix * beta[:, None]
        self.matrix[self.matrix <= threshold] = epsilon

    def apply(
        self, metric: str, save: bool = False, path: None | Path = None
    ) -> pd.DataFrame:
        res = []
        func = getattr(import_module("algo"), metric)
        # TODO: apply_along_axis or something
        for i in range(len(self.matrix) - 1):
            res.append(func(self.matrix[i : i + 2]))
        res_df = (
            pd.DataFrame({metric: res}).reset_index().rename(columns={"index": "date"})
        )
        if save:
            write(path, (res_df, metric))
        return res_df

    def delete(self, ix, axis):
        self.matrix = np.delete(self.matrix, obj=ix, axis=axis)

    def normalize(self):
        self.normalized = True
        self.matrix = (self.matrix.T / self.matrix.sum(axis=1)).T

    def create_dvr(self, mean=False):
        if self.normalized:
            raise ValueError("Cannot create the DVR from normalized frequency data")
        if mean:
            self.dvr = self.normalized_average_weight()
        else:
            self.dvr = self.normalized_weight()

    def normalized_average_weight(self) -> np.ndarray:
        average_weight = bn.nanmean(self.matrix, axis=0)
        return average_weight / average_weight.sum()

    def normalized_weight(self) -> np.ndarray:
        return self.matrix.sum(axis=0) / self.matrix.sum()

    def moving_average(self, window: int) -> np.array:
        max_ = bn.nanmax(self.matrix, axis=1)
        min_ = bn.nanmin(self.matrix, axis=1)
        ma = bn.move_mean(bn.nanmean(self.matrix, axis=1), window=window, min_count=1)
        return pd.DataFrame({"ma": ma, "max": max_, "min": min_}).reset_index()


class Corpus:
    def __init__(
        self,
        freq: pd.DataFrame | None = None,
        document_cat: pd.Series | pd.DatetimeIndex | None = None,
        element_cat: pd.Series | None = None,
        name: str | None = None,
    ):
        if (
            isinstance(freq, type(None))
            and isinstance(document_cat, type(None))
            and isinstance(element_cat, type(None))
        ):
            raise ValueError(
                "Either use a frequency dataframe or two series, one of document ids and one of elements"
            )
        elif isinstance(freq, pd.DataFrame):
            self.freq = freq
            document_cat = freq["document"]
            element_cat = freq["element"]
        self.document_cat = pd.Categorical(document_cat, ordered=True).dtype
        self.element_cat = pd.Categorical(element_cat, ordered=True).dtype
        if name:
            self.name = name

    def __len__(self):
        """Number of documents"""
        return len(self.matrix.matrix)

    def update_documents(self, document):
        self.document_cat = pd.CategoricalDtype(
            self.document_cat.categories[
                ~self.document_cat.categories.isin([document])
            ],
            ordered=True,
        )

    def code_to_cat(self, code: str, what="document") -> int:
        return getattr(self, f"{what}_cat").categories[code]

    def pivot(self, freq: pd.DataFrame | None = None) -> Matrix:
        if hasattr(self, "freq"):
            freq = self.freq
        d = freq["document"].astype(self.document_cat)
        e = freq["element"].astype(self.element_cat)
        idx = np.array([d.cat.codes, e.cat.codes]).T
        matrix = np.zeros(
            (len(d.cat.categories), len(e.cat.categories)), dtype="float64"
        )
        matrix[idx[:, 0], idx[:, 1]] = freq["frequency_in_document"]
        return Matrix(matrix[min(d.cat.codes) : max(d.cat.codes) + 1])

    def create_dvr(
        self, equally_weighted: bool = False, matrix: None | Matrix = None
    ) -> pd.DataFrame:
        if not matrix:
            self.matrix = self.pivot(self.freq)
            matrix = self.matrix
        matrix.create_dvr(mean=equally_weighted)
        dvr = (
            pd.DataFrame(
                {
                    "element": self.element_cat.categories,
                    "global_weight": matrix.dvr,
                }
            )
            .reset_index()
            .rename(columns={"index": "element_code"})
            .sort_values("global_weight", ascending=False)
            .reset_index(drop=True)
        )
        return dvr[["element", "global_weight"]]

    def create_signatures(
        self,
        epsilon: float,
        most_significant: int | None = 30,
        sig_length: int | None = 500,
        prevelent: int | None = None,
    ) -> List[pd.DataFrame] | Tuple[List[pd.DataFrame]]:
        if sig_length == 0:
            sig_length = None
        if not hasattr(self, "matrix"):
            raise AttributeError("Please create dvr before creating signatures.")
        if not self.matrix.normalized:
            self.matrix.normalize()
        if prevelent:
            temporary_array = np.count_nonzero(self.matrix.matrix, axis=0) <= prevelent
        self.matrix.epsilon_modification(epsilon)
        self.distance_matrix = Matrix(
            KLD_distance_overused(self.matrix.matrix, self.matrix.dvr)
        )
        if prevelent:
            self.prevelent_matrix = self.distance_matrix.matrix.copy()
            self.prevelent_matrix[
                temporary_array & (self.distance_matrix.matrix < 0)
            ] = 0
        distances_df = pd.DataFrame(
            (self.prevelent_matrix if prevelent else self.distance_matrix.matrix),
            index=self.document_cat.categories,
            columns=self.element_cat.categories,
        )
        signatures = [
            sig.loc[sig.abs().sort_values(ascending=False).index].head(sig_length)
            for _, sig in distances_df.iterrows()
        ]
        res = [signatures]
        if most_significant:
            sort = np.argsort(
                np.abs(self.distance_matrix.matrix).sum(axis=0), kind="stable"
            )[-most_significant:][::-1]
            max_distances_df = distances_df.iloc[:, sort]
            max_distances = [dist for _, dist in max_distances_df.iterrows()]
            res.append(max_distances)
        if prevelent:
            print("yeassss qveen")
            res.append(temporary_array)
        return tuple(res)


def sockpuppet_distance(
    corpus1: Corpus, corpus2: Corpus, res: Literal["table", "matrix"] = "table"
) -> pd.DataFrame:
    cc = []
    for c in [corpus1, corpus2]:
        if hasattr(c, "prevelent_matrix"):
            x = np.where(
                c.prevelent_matrix > 0, c.prevelent_matrix + 1, c.prevelent_matrix - 1
            )
        else:
            x = np.where(
                c.distance_matrix.matrix > 0,
                c.distance_matrix.matrix + 1,
                c.distance_matrix.matrix - 1,
            )
        cc.append(x)
    df = pd.DataFrame(
        cdist(cc[0], cc[1], metric="cityblock"),
        index=corpus1.document_cat.categories,
        columns=corpus2.document_cat.categories,
    )
    df /= df.max().max()
    if res == "table":
        df = (
            df.rename_axis(index=getattr(corpus1, "name", "Corpus 1"))
            .melt(ignore_index=False, var_name=getattr(corpus1, "name", "Corpus 2"))
            .reset_index()
        )
    return df
