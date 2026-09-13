#!/usr/bin/env python3
"""
Module for setting the Timestamp column as the DataFrame index.
"""


def index(df):
    """
    Sets the Timestamp column as the index.

    Args:
        df: pandas DataFrame containing a Timestamp column.

    Returns:
        The modified pandas DataFrame.
    """
    return df.set_index("Timestamp")
