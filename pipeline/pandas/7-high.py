#!/usr/bin/env python3
"""
Module for sorting a DataFrame by High price.
"""


def high(df):
    """
    Sorts a DataFrame by High price in descending order.

    Args:
        df: pandas DataFrame containing a High column.

    Returns:
        The sorted pandas DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
