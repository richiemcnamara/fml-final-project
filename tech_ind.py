import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def price_SMA_ratio(df, n, sym):
    """Input DataFrame with the column [Sym], n for the SMA, and sym for the ticker
       Returns DataFrame with the column [Price to SMA-{n} Ratio]"""

    df_ind = df.copy()

    df_ind[f"Price to SMA-{n} Ratio"] = df_ind[f"{sym}"] / (df[f"{sym}"].rolling(window=n).mean())
    del df_ind[f"{sym}"]

    return df_ind


def boll_bands(df, n, Sym):
    """Helper for calculating the BB %
       Input Dataframe with the column [Sym]
       Returns DataFrame with the column [Sym, SMA-n, Lower Band, Upper Band]"""
    
    df_bb = df.copy()

    df_bb[f"SMA-{n}"] = df[f"{Sym}"].rolling(window=n).mean()
    df_bb[f"SD-{n}"] = df[f"{Sym}"].rolling(window=n).std()

    df_bb["Upper Band"] = df_bb[f"SMA-{n}"] + (2 * df_bb[f"SD-{n}"])
    df_bb["Lower Band"] = df_bb[f"SMA-{n}"] - (2 * df_bb[f"SD-{n}"])

    del df_bb[f"SD-{n}"]

    return df_bb


def boll_bands_percent(df, n, Sym):
    """Input DataFrame with the columns [Sym, SMA-n, Lower Band, Upper Band] and the n for the BB
       Returns DataFrame with the column [BB %]"""
    
    df_ind = boll_bands(df, n, Sym)

    df_ind["BB %"] = ((df_ind[f"{Sym}"] - df_ind["Lower Band"]) / (df_ind["Upper Band"] - df_ind["Lower Band"])) * 100

    del df_ind[f"{Sym}"]
    del df_ind["Lower Band"]
    del df_ind["Upper Band"]

    return df_ind


def stochastic_oscillator(df, Sym):
    """Input DataFrame with the columns [Sym, High, Low]
       Sym is the close price for that symbol
       Returns DataFrame with the column [%K, %D]"""

    df_ind = df.copy()

    df_ind["Low-14"] = df_ind["Low"].rolling(14).min()
    df_ind["High-14"] = df_ind["High"].rolling(14).max()

    df_ind["%K"] = ((df_ind[f"{Sym}"] - df_ind["Low-14"]) * 100) / (df_ind["High-14"] - df_ind["Low-14"])
    df_ind["%D"] = df_ind["%K"].rolling(window=3).mean()

    del df_ind[f"{Sym}"]
    del df_ind["High"]
    del df_ind["Low"]
    del df_ind["Low-14"]
    del df_ind["High-14"]

    return df_ind
