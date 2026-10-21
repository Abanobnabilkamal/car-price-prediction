
import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import RESULTS_PATH


st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Performance Comparison")
st.write("Compare the performance of the trained regression models.")

# Load saved results
if not RESULTS_PATH.exists():
    st.error("Comparison results not found. Please run train.py first.")
    st.stop()

results_df = pd.read_csv(RESULTS_PATH)

if results_df.empty:
    st.warning("The comparison results file is empty.")
    st.stop()

# Display metrics
col1, col2, col3 = st.columns(3)

best_model = results_df.sort_values("CV RMSE").iloc[0]

col1.metric("Best Model", best_model["Model"])
col2.metric("Best CV RMSE", f'{best_model["CV RMSE"]:,.2f}')
col3.metric("Best Test R²", f'{best_model["Test R2"]:.4f}')

st.divider()

# RMSE Comparison
st.subheader("RMSE Comparison")

fig_rmse = px.bar(
    results_df,
    x="Model",
    y="Test RMSE",
    color="Model",
    title="Test RMSE by Model",
    text_auto=".2s",
)

st.plotly_chart(fig_rmse, width="stretch")

# MAE Comparison
st.subheader("MAE Comparison")

fig_mae = px.bar(
    results_df,
    x="Model",
    y="Test MAE",
    color="Model",
    title="Test MAE by Model",
    text_auto=".2s",
)

st.plotly_chart(fig_mae, width="stretch")

# R² Score Comparison
st.subheader("R² Score Comparison")

fig_r2 = px.bar(
    results_df,
    x="Model",
    y="Test R2",
    color="Model",
    title="Test R² Score by Model",
    text_auto=".3f",
)

st.plotly_chart(fig_r2, width="stretch")

# Cross-Validation vs Test RMSE
st.subheader("Cross-Validation vs Test RMSE")

rmse_data = results_df.melt(
    id_vars="Model",
    value_vars=["CV RMSE", "Test RMSE"],
    var_name="Evaluation",
    value_name="RMSE",
)

fig_cv = px.bar(
    rmse_data,
    x="Model",
    y="RMSE",
    color="Evaluation",
    barmode="group",
    title="CV RMSE vs Test RMSE",
    text_auto=".2s",
    labels={
        "RMSE": "RMSE",
        "Evaluation": "Metric",
        "Model": "Model",
    },
)

fig_cv.update_layout(bargap=0.3)

st.plotly_chart(fig_cv, width="stretch")

# Complete Results
st.subheader("Complete Results")

st.dataframe(
    results_df.drop(columns=["Best Params"], errors="ignore"),
    width="stretch",
    hide_index=True,
)


