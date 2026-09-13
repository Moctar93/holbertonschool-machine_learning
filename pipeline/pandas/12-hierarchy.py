#!/usr/bin/env python3
"""
Module for rearranging and concatenating DataFrames.
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearranges and concatenates Coinbase and Bitstamp data.

    Args:
        df1: pandas DataFrame containing Coinbase data.
        df2: pandas DataFrame containing Bitstamp data.

    Returns:
        The concatenated pandas DataFrame in chronological order.
    """
    df1 = index(df1)
    df2 = index(df2)

    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    df = pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )

    df = df.swaplevel(0, 1)
    df = df.sort_index()

    return df
