#!/usr/bin/env python3


def array(df):
    """
    Selects the last 10 rows of High and Close
    and converts them to a numpy.ndarray.

    Args:
        df: pandas DataFrame

    Returns:
        numpy.ndarray
    """
    return df[["High", "Close"]].tail(10).to_numpy()
