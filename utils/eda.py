# utils/eda.py

import pandas as pd
import plotly.express as px

def show_head(df: pd.DataFrame, rows: int = 5):
    return df.head(rows)

def show_describe(df: pd.DataFrame):
    return df.describe()

def show_dtypes(df: pd.DataFrame):
    return df.dtypes

def show_corr_heatmap(df: pd.DataFrame):
    """
    Returns a Plotly Figure for the correlation heatmap.
    """
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    return fig

def show_missing_values(df: pd.DataFrame):
    return df.isnull().sum()

def show_histogram(df: pd.DataFrame, column: str):
    fig = px.histogram(df, x=column, title=f"Histogram of {column}")
    return fig

def show_boxplot(df: pd.DataFrame, column: str):
    fig = px.box(df, y=column, title=f"Boxplot of {column}")
    return fig

def show_value_counts_bar(df: pd.DataFrame, column: str):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count", title=f"Value Counts of {column}")
    return fig