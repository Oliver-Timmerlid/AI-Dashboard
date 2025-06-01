# utils/model_router.py

from utils import regression, kMeans

def get_model_ui(model_name, df):
    if model_name == "Regression":
        return regression.model_ui(df)
    elif model_name == "KMeans Clustering":
        return kMeans.model_ui(df)
    else:
        return "Model support not yet available."
