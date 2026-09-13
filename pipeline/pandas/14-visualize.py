#!/usr/bin/env python3
"""
Module for visualizing Bitcoin data.
"""

import pandas as pd
import matplotlib.pyplot as plt


def visualize(df):
    """
    Cleans, transforms and plots Bitcoin data from 2017 onwards.

    Args:
        df: pandas DataFrame containing Bitcoin data.

    Returns:
        The transformed pandas DataFrame before plotting.
    """
    df = df.drop(columns=["Weighted_Price"])

    df = df.rename(columns={"Timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], unit="s")
    df = df.set_index("Date")

    df["Close"] = df["Close"].ffill()

    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    df = df.loc["2017":]

    df = df.resample("D").agg({
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum"
    })

    df.plot()

    return df
