import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import DATA_PATH
from src.preprocessing import load_and_clean_data


st.set_page_config(
    page_title="Data Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Car Price Data Analysis")
st.write("Explore the dataset and investigate factors related to car prices.")

# Load and clean dataset
df = load_and_clean_data(DATA_PATH)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Cars", f"{len(df):,}")
col2.metric("Average Price", f"{df['Price'].mean():,.0f}")
col3.metric("Median Price", f"{df['Price'].median():,.0f}")
col4.metric("Manufacturers", f"{df['Manufacturer'].nunique():,}")

st.divider()

# Price distribution
st.subheader("1. Price Distribution")

fig_price = px.histogram(
    df,
    x="Price",
    nbins=50,
    title="Distribution of Car Prices",
    labels={"Price": "Price"},
)

st.plotly_chart(fig_price, use_container_width=True)

# Manufacturer analysis
st.subheader("2. Price by Manufacturer")

top_n = st.slider(
    "Number of manufacturers to display",
    min_value=5,
    max_value=20,
    value=10,
)

top_manufacturers = df["Manufacturer"].value_counts().head(top_n).index

manufacturer_df = df[df["Manufacturer"].isin(top_manufacturers)]

manufacturer_prices = (
    manufacturer_df.groupby("Manufacturer", observed=True)["Price"]
    .median()
    .sort_values(ascending=False)
    .reset_index()
)

fig_manufacturer = px.bar(
    manufacturer_prices,
    x="Manufacturer",
    y="Price",
    title="Median Car Price by Manufacturer",
    labels={"Price": "Median Price"},
)

st.plotly_chart(fig_manufacturer, use_container_width=True)

# Mileage vs Price
st.subheader("3. Mileage vs Price")

fig_mileage = px.scatter(
    df,
    x="Mileage",
    y="Price",
    color="Fuel type",
    title="Mileage vs Car Price",
    opacity=0.6,
    hover_data=["Manufacturer", "Model", "Prod. year"],
)

st.plotly_chart(fig_mileage, use_container_width=True)

# Production year analysis
st.subheader("4. Price by Production Year")

year_df = (df.groupby("Prod. year", as_index=False)["Price"].median().sort_values("Prod. year"))

fig_year = px.line(
    year_df,
    x="Prod. year",
    y="Price",
    markers=True,
    title="Median Car Price by Production Year",
    labels={
        "Prod. year": "Production Year",
        "Price": "Median Price",
    },
)

st.plotly_chart(fig_year, use_container_width=True)

# Dataset preview
with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)