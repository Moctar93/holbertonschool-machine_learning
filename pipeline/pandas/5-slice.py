#!/usr/bin/env python3
"""
Module for slicing a pandas DataFrame.
"""


def slice(df):
    """
    Extracts selected columns and every 60th row.

    Args:
        df: pandas DataFrame containing High, Low, Close,
            and Volume_(BTC) columns.

    Returns:
        The sliced pandas DataFrame.
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
