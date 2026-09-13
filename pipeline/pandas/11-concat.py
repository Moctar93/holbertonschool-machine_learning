#!/usr/bin/env python3
"""
Module for concatenating two DataFrames.
"""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates selected Bitstamp data with Coinbase data.

    Args:
        df1: pandas DataFrame containing Coinbase data.
        df2: pandas DataFrame containing Bitstamp data.

    Returns:
        The concatenated pandas DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[:1417411920]

    return pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )
