#!/usr/bin/env python3
"""
Module for computing descriptive statistics.
"""

import pandas as pd


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Args:
        df: pandas DataFrame containing a Timestamp column.

    Returns:
        A pandas DataFrame containing descriptive statistics.
    """
    return df.drop(columns=["Timestamp"]).describe()
