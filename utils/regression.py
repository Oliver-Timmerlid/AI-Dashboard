# utils/regression.py

import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

def train_regression_model(df, features, target, test_size):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

def visualize_regression(X_test, y_test, y_pred):
    fig = go.Figure()

    if X_test.shape[1] == 1:
        feature_name = X_test.columns[0]
        x_vals = X_test[feature_name]
        fig.add_trace(go.Scatter(x=x_vals, y=y_test, mode='markers', name='Actual'))
        fig.add_trace(go.Scatter(x=x_vals, y=y_pred, mode='lines', name='Prediction'))
        fig.update_layout(title="Regression Line", xaxis_title=feature_name, yaxis_title="Target")
    else:
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Prediction'))
        fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Perfect (y=x)', line=dict(dash='dash')))
        fig.update_layout(title="Prediction vs Actual", xaxis_title="Actual", yaxis_title="Prediction")
    
    return fig

def model_ui(df):
    st.subheader("Regression")

    target_col = st.selectbox("Select target (y)", df.columns, key="reg_target")
    feature_cols = st.multiselect("Select features (X)", [col for col in df.columns if col != target_col], key="reg_features")
    test_size = st.slider("Train/Test Split", 0.1, 0.5, 0.2, key="reg_split")

    if st.button("Train regression model"):
        if feature_cols and target_col:
            model, X_test, y_test, y_pred = train_regression_model(df, feature_cols, target_col, test_size)
            st.success("Model trained!")
            st.write("R²-score:", model.score(df[feature_cols], df[target_col]))
            st.subheader("Visualization")
            st.plotly_chart(visualize_regression(X_test, y_test, y_pred))
