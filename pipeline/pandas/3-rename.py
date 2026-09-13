#!/usr/bin/env python3
import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime,
    converts timestamps to datetime values,
    and keeps only Datetime and Close columns.

    Args:
        df: pandas DataFrame

    Returns:
        The modified DataFrame
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]

    return df
