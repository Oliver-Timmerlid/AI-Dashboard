import pandas as pd
import streamlit as st

def interactive_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    changes = []

    st.markdown("### 🧹 Välj transformationsalternativ")

    # Datumomvandling
    date_cols = st.multiselect("📆 Välj kolumner att tolka som datum", df.columns)
    for col in date_cols:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col])
            with st.expander(f"📅 Extrahera komponenter från {col}"):
                extract_year = st.checkbox(f"År från {col}", value=True, key=f"{col}_year")
                extract_month = st.checkbox(f"Månad från {col}", value=True, key=f"{col}_month")
                extract_day = st.checkbox(f"Dag från {col}", value=False, key=f"{col}_day")
                extract_weekday = st.checkbox(f"Veckodag från {col}", value=False, key=f"{col}_weekday")
                extract_ordinal = st.checkbox(f"Ordinal från {col}", value=True, key=f"{col}_ordinal")

            if extract_year:
                df_clean[f"{col}_year"] = df_clean[col].dt.year
            if extract_month:
                df_clean[f"{col}_month"] = df_clean[col].dt.month
            if extract_day:
                df_clean[f"{col}_day"] = df_clean[col].dt.day
            if extract_weekday:
                df_clean[f"{col}_weekday"] = df_clean[col].dt.weekday
            if extract_ordinal:
                df_clean[f"{col}_ordinal"] = df_clean[col].map(pd.Timestamp.toordinal)

            df_clean.drop(columns=[col], inplace=True)
            changes.append(f"📆 {col}: Datumkomponenter extraherade")

        except Exception as e:
            st.warning(f"Kunde inte tolka {col} som datum: {e}")

    # One-hot encoding
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_choices = st.multiselect("🏷️ Välj kategoriska kolumner för one-hot encoding", cat_cols)

    if st.button("Utför one-hot encoding"):
        df_clean = pd.get_dummies(df_clean, columns=cat_choices)
        changes.append(f"🏷️ One-hot encoded: {', '.join(cat_choices)}")

    if changes:
        st.info("Följande förändringar gjordes:\n" + "\n".join(changes))
    else:
        st.info("Inga förändringar gjorda än.")

    return df_clean
