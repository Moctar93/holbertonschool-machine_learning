#!/usr/bin/env python3
"""
Module for cleaning and filling missing values in a DataFrame.
"""


def fill(df):
    """
    Removes Weighted_Price and fills missing values.

    Args:
        df: pandas DataFrame.

    Returns:
        The modified pandas DataFrame.
    """
    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].ffill()

    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df 
