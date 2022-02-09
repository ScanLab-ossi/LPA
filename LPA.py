from itertools import combinations_with_replacement as cwr
from itertools import product
from typing import Optional
from matplotlib.pyplot import isinteractive

import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from helpers import timing


class LPA:
    def __init__(
        self, dvr: pd.DataFrame, categories: int = 1000, epsilon_frac: int = 2
    ):
        self.dvr = dvr
        self.epsilon = 1 / (len(dvr) * epsilon_frac)
        self.categories = categories

    @staticmethod
    def create_dvr(df: pd.DataFrame) -> pd.DataFrame:
        """Creates the DVR table of the domain"""
        dvr = (
            df.groupby("element", as_index=False)
            .sum()
            .sort_values(by="frequency_in_category", ascending=False)
        )
        dvr["global_weight"] = dvr["frequency_in_category"] / sum(
            dvr["frequency_in_category"]
        )
        return dvr.reset_index(drop=True)

    def create_pvr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a vector for every category in the domain"""
        df["local_weight"] = df["frequency_in_category"] / df.groupby("category")[
            "frequency_in_category"
        ].transform("sum")
        return df

    @staticmethod
    def KLD(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Kullback-Leibler distance.
        P represents the data, the observations, or a measured probability distribution.
        Q represents instead a theory, a model, a description or an approximation of P.
        """

        return (P - Q) * (ma.log(P) - ma.log(Q))

    def extended_freq_vec(
        self, vec: pd.DataFrame, vec_lengths: np.ndarray, missing: np.ndarray
    ) -> pd.DataFrame:
        # TODO: rename
        betas = [
            item
            for sublist in [
                times * [(1 - missing * self.epsilon)[i]]
                for i, times in enumerate(vec_lengths)
            ]
            for item in sublist
        ]
        vec["local_weight"] = vec["local_weight"] * pd.Series(betas)
        return vec

    def betas(self, pvr: pd.DataFrame) -> pd.DataFrame:
        pvr_lengths = (
            pvr["category"].drop_duplicates(keep="last").index
            - pvr["category"].drop_duplicates(keep="first").index
            + 1
        ).to_numpy()
        missing = len(self.dvr) - pvr_lengths
        return self.extended_freq_vec(pvr, pvr_lengths, missing)

    def get_missing(self, length: int) -> pd.DataFrame:
        missing = self.dvr.loc[: length - 1, "element"].copy().to_frame()
        missing["KL"] = self.KLD(
            self.dvr.loc[: length - 1, "global_weight"].copy().to_numpy(), self.epsilon
        )
        missing["missing"] = True
        return missing

    def create_signatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the raw data and creates signatures for every category in the domain.
        `epsilon_frac` defines the size of epsilon, default is 1/(corpus size * 2)
        `sig_length` defines the length of the signature, default is 500"""
        vecs = self.betas(self.create_pvr(df)).pivot_table(
            values="local_weight", index="element", columns="category"
        )
        dvr_array = (
            self.dvr[self.dvr["element"].isin(vecs.index)]
            .sort_values("element")["global_weight"]
            .to_numpy()
        )
        vecs_array = vecs.fillna(0).to_numpy().T
        distances = (
            pd.DataFrame(
                self.KLD(dvr_array, vecs_array),
                index=vecs.columns,
                columns=vecs.index,
            )
            .stack()
            .reset_index()
            .rename(columns={0: "KL"})
        )
        return distances

    def diminishing_return(
        self, sigs: pd.DataFrame, sig_length: int = 500
    ) -> pd.DataFrame:
        sigs["missing"] = False
        missing = self.get_missing(sig_length)
        categories = sigs["category"].drop_duplicates().reset_index(drop=True)
        merged = pd.merge(categories, missing, how="cross")
        sigs = (
            sigs.append(merged)
            .sort_values(["category", "KL"], ascending=[True, False])
            .groupby("category")
            .head(sig_length)
            .reset_index(drop=True)
        )
        return sigs

    def create_and_cut(self, df: pd.DataFrame, sig_length: int = 500) -> pd.DataFrame:
        sigs = self.create_signatures(df)
        return self.diminishing_return(sigs, sig_length)

    def distance_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        sigs = self.create_signatures(df)
        return sigs.groupby("category").sum()

    @timing
    def sockpuppet_distance(
        self, signatures1: pd.DataFrame, signatures2: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Returns size*size df
        """
        # TODO: triu
        categories1 = signatures1["category"].drop_duplicates()
        categories2 = signatures2["category"].drop_duplicates()
        pivot = signatures1.append(signatures2).pivot_table(
            values="KL", index="category", columns="element", fill_value=0
        )
        XA = pivot.filter(categories1, axis="index")  # .to_numpy()
        XB = pivot.filter(categories2, axis="index")  # .to_numpy()
        df = pd.DataFrame(
            cdist(XA, XB, metric="cityblock"), index=categories1, columns=categories2
        )
        return df


class IterLPA(LPA):
    def __init__(self, blocks, size, dvr):
        super().__init__(dvr=dvr)
        self.blocks = blocks
        self.size = size
        self._range = range(0, self.size * self.blocks, self.size)

    @staticmethod
    def create_dvr(df):
        """Creates the DVR table of the domain"""
        dvr = (
            df.groupby("element", as_index=False)
            .sum()
            .sort_values(by="frequency_in_category", ascending=False)
        )
        dvr["global_weight"] = dvr["frequency_in_category"] / sum(
            dvr["frequency_in_category"]
        )
        return dvr

    def _shorthand(self, stage):
        shorthand = {"frequency": "freq", "signatures": "sigs"}
        return shorthand[stage]

    def _grid(self, symmetric=True):
        if symmetric:
            return list(cwr(self._range, r=2))
        else:
            return list(product(self._range, r=2))

    def iter_dvr(self):
        l = []
        for i in self._range:
            l.append(pd.read_csv(f"data/freq/frequency_{i}.csv"))
        self.create_dvr(pd.concat(l)).to_csv("dvr1.csv", index=False)

    def run_sockpuppets(self):
        for i, j in self._grid():
            sigs1 = self.create_and_cut(pd.read_csv(f"data/freq/frequency_{i}.csv"))
            sigs2 = self.create_and_cut(pd.read_csv(f"data/freq/frequency_{j}.csv"))
            self.sockpuppet_distance(sigs1, sigs2).to_csv(
                f"data/sockpuppets/sp_{i}_{j}.csv"
            )
            print(f"finished spd for blocks {i}, {j}")

    def PCA(self):
        df = pd.read_csv(f"data/sockpuppets/sp_{i}_{j}.csv").set_index("category")
        df = StandardScaler().fit_transform(df)
        pca = PCA(n_components=2)
        pcdf = pca.fit_transform(df)
