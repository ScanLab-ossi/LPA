from __future__ import annotations

from copy import copy
from importlib import import_module
from itertools import combinations_with_replacement as cwr
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import lambertw
from sklearn import decomposition as skd
from sklearn.preprocessing import StandardScaler

from algo import symmetrized_KLD, entropy
from helpers import write


class Matrix:
    def __init__(self, matrix: np.array):
        self.matrix = matrix
        self.normalized = False

    def __bool__(self):
        return True if hasattr(self, "matrix") else False

    def epsilon_modification(
        self,
        epsilon: float | None = None,
        lambda_: float | int = 1,
        threshold: float = 0,
    ):
        if not epsilon:
            if epsilon == 0:
                return
            raise ValueError("Epsilon must be provided")
            # epsilon = self._get_epsilon(lambda_)
            # if epsilon == 0:
            #     return
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

    def create_dvr(self):
        if self.normalized:
            raise ValueError("Cannot create the DVR from normalized frequency data")
        self.dvr = self.normalized_weight()

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

    def current(self, m=True):
        if hasattr(self, "signature_matrix"):
            curr = self.signature_matrix
        elif hasattr(self, "distance_matrix"):
            curr = self.distance_matrix
        return curr.matrix if m else curr

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

    def create_dvr(self, matrix: None | Matrix = None) -> pd.DataFrame:
        if not matrix:
            self.matrix = self.pivot(self.freq)
            matrix = self.matrix
        matrix.create_dvr()
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

    def _signature_matrix(self, sig_length, distances_df):
        # annuls all values that shouldn't appear in the signatures
        self.signature_matrix = Matrix(self.current().copy())  # copy?
        if sig_length:
            argsort = np.argsort(np.abs(self.signature_matrix.matrix), axis=1)
            indices = argsort[:, -sig_length:]
            p = np.zeros_like(self.signature_matrix.matrix)
            for i in range(p.shape[0]):
                p[i, indices[i]] = self.signature_matrix.matrix[i, indices[i]]
            self.signature_matrix.matrix = p
        signatures = [
            sig[1][self.signature_matrix.matrix[i] != 0].sort_values(
                key=lambda x: abs(x), ascending=False
            )
            for i, sig in enumerate(distances_df.iterrows())
        ]
        return signatures

    def create_signatures(
        self,
        epsilon: float | None = None,
        sig_length: int | None = 500,
        distance: str = "KLDe",
    ) -> List[pd.DataFrame] | Tuple[List[pd.DataFrame]]:
        """
        most_significant: checks which elements had the largest distance altogether and returns a dataframe consisting only of those distances, sorted
        """
        if sig_length == 0:
            sig_length = None
        if not hasattr(self, "matrix"):
            raise AttributeError("Please create dvr before creating signatures.")
        if not self.matrix.normalized:
            self.matrix.normalize()
        if distance == "KLDe":
            self.matrix.epsilon_modification(epsilon)
        dm = symmetrized_KLD(self.matrix.matrix, self.matrix.dvr)
        self.distance_matrix = Matrix(dm)
        distances_df = pd.DataFrame(
            self.current(),
            index=self.document_cat.categories,
            columns=self.element_cat.categories,
        )
        res = self._signature_matrix(sig_length, distances_df)
        return res


def sockpuppet_distance(
    corpus1: Corpus,
    corpus2: Corpus,
    res: Literal["table", "matrix"] = "table",
    heuristic: bool = True,
) -> pd.DataFrame:
    matrices = []
    for c in [corpus1, corpus2]:
        matrix = copy(c.signature_matrix)
        matrix.matrix = matrix.matrix[:, ~np.all(matrix.matrix == 0, axis=0)]
        if heuristic:
            matrix.matrix[matrix.matrix > 0] += 1
            matrix.matrix[matrix.matrix < 0] -= 1
        matrices.append(matrix.matrix)
    c1n = getattr(corpus2, "name", "Corpus 1")
    c2n = getattr(corpus1, "name", "Corpus 2")
    cdist_ = cdist(matrices[0], matrices[1], metric="cityblock")
    cdist_[np.triu_indices(len(cdist_), k=1)] = np.nan
    df = pd.DataFrame(
        cdist_,
        index=corpus1.document_cat.categories,
        columns=corpus2.document_cat.categories,
    )
    if c1n == c2n:
        c2n = c1n + " "
    df = (
        df.rename_axis(index=c1n)
        .melt(ignore_index=False, var_name=c2n)
        .dropna()
        .reset_index()
    )
    df["value"] /= df["value"].max()
    if res == "matrix":
        df = df.pivot(index=c1n, columns=c2n, values="value").fillna(0)
        df = df + df.T
    return df


def PCA(sockpuppet_matrix, n_components: int = 2):
    """
    Creates a PCA object and returns it, as well as the explained variance ratio.
    """
    scaler = StandardScaler()
    sockpuppet_matrix = scaler.fit_transform(sockpuppet_matrix)
    scaled_matrix = scaler.fit_transform(sockpuppet_matrix)
    pca = skd.PCA(n_components=n_components)
    pca.fit(scaled_matrix)
    res = pca.transform(scaled_matrix)
    return res, pca.explained_variance_ratio_
