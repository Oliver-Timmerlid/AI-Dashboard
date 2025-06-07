import streamlit as st
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer
from utils.feature_cleaner import interactive_cleaner

from utils.model_router import get_model_ui

st.set_page_config(page_title="AI Dashboard", layout="wide")
st.title("AI Dashboard")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded!")

    st.subheader("🛠 Feature Cleaning (Interaktiv)")
    df = interactive_cleaner(raw_df)

    @st.cache_resource
    def get_pyg_renderer(data: pd.DataFrame) -> "StreamlitRenderer":
        return StreamlitRenderer(data, spec="./gw_config.json", spec_io_mode="rw")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Interaktiv EDA (PyGWalker)")
        renderer = get_pyg_renderer(df)
        renderer.explorer()

    with col2:
        st.subheader("🤖 Modellval")
        model_choice = st.selectbox("Välj modell", ["Regression", "KMeans Clustering"])
        get_model_ui(model_choice, df)

