import streamlit as st
import pandas as pd
from utils.eda import show_head, show_describe, show_dtypes
from utils.model_router import get_model_ui

st.set_page_config(page_title="AI Dashboard MVP", layout="wide")
st.title(" AI Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded!")

    st.subheader("🔍 Simple EDA")
    if st.checkbox("Show first rows"):
        st.dataframe(show_head(df))
    if st.checkbox("Show describe()"):
        st.write(show_describe(df))
    if st.checkbox("Show data types"):
        st.write(show_dtypes(df))

    st.subheader("Model Selection")
    model_choice = st.selectbox("Choose model", ["Regression", "KMeans Clustering"])

    get_model_ui(model_choice, df)
