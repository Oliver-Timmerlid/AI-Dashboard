import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans

def model_ui(df):
    st.subheader("KMeans Clustering")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    max_k = st.slider("Max number of clusters for elbow", 2, 10, 5)

    if st.button("Show Elbow Plot"):
        distortions = []
        for k in range(1, max_k+1):
            model = KMeans(n_clusters=k, n_init='auto')
            model.fit(df[numeric_cols])
            distortions.append(model.inertia_)

        fig = px.line(
            x=list(range(1, max_k+1)),
            y=distortions,
            markers=True,
            labels={"x": "Number of clusters (K)", "y": "Inertia"},
            title="Elbow Method"
        )
        st.plotly_chart(fig)

    n_clusters = st.number_input("Select number of clusters (K)", min_value=1, value=3)

    if st.button("Run KMeans"):
        model = KMeans(n_clusters=n_clusters, n_init='auto')
        clustered_df = df.copy()
        clustered_df["Cluster"] = model.fit_predict(df[numeric_cols])
        
        # Save in session_state
        st.session_state["kmeans_model"] = model
        st.session_state["clustered_df"] = clustered_df
        st.session_state["numeric_cols"] = numeric_cols
        st.success(f"KMeans done – {n_clusters} clusters")

    # If the model already exists – show visualization
    if "clustered_df" in st.session_state:
        st.subheader("Cluster Visualization")
        clustered_df = st.session_state["clustered_df"]
        numeric_cols = st.session_state["numeric_cols"]

        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("X-axis", numeric_cols, index=0, key="xaxis")
            y_axis = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_axis], index=0, key="yaxis")

            fig = px.scatter(
                clustered_df, x=x_axis, y=y_axis, color="Cluster",
                title="KMeans Clusters (2D)", opacity=0.7
            )
            st.plotly_chart(fig)

        elif len(numeric_cols) == 1:
            st.warning("Only one numeric column – hard to visualize clusters.")
        else:
            st.error("No numeric columns to cluster.")

        with st.expander("Show data table with clusters"):
            st.dataframe(clustered_df.head())
