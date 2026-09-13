#!/usr/bin/env python3
"""
Module for renaming and converting the Timestamp column.
"""

import pandas as pd


def rename(df):
    """
    Renames Timestamp to Datetime and converts timestamps to datetime.

    Args:
        df: pandas DataFrame containing a Timestamp column.

    Returns:
        The modified pandas DataFrame.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
