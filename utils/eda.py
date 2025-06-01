# utils/eda.py

import pandas as pd

def show_head(df: pd.DataFrame, rows: int = 5):
    return df.head(rows)

def show_describe(df: pd.DataFrame):
    return df.describe()

def show_dtypes(df: pd.DataFrame):
    return df.dtypes

