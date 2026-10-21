import streamlit as st

from predict import predict_price


# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered",
)

# Page title
st.title("🚗 Car Price Prediction")
st.write("Enter the car details to predict its price.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    manufacturer = st.text_input("Manufacturer", value="Toyota")
    model = st.text_input("Model", value="Prius")
    prod_year = st.number_input(
        "Production Year", min_value=1900, max_value=2026, value=2018
    )
    engine_volume = st.number_input(
        "Engine Volume (L)", min_value=0.1, max_value=10.0, value=2.0, step=0.1
    )
    mileage = st.number_input(
        "Mileage (km)", min_value=0, value=120000, step=1000
    )
    cylinders = st.number_input(
        "Cylinders", min_value=1, max_value=16, value=4
    )
    levy = st.number_input("Levy", min_value=0.0, value=500.0)
    airbags = st.number_input(
        "Airbags", min_value=0, max_value=20, value=6
    )

with col2:
    category = st.selectbox(
        "Category",
        ["Sedan", "Jeep", "Hatchback", "Coupe", "Universal", "Other"],
    )

    leather = st.selectbox("Leather Interior", ["Yes", "No"])

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "Hybrid", "Electric", "LPG", "CNG", "Hydrogen"],
    )

    gear_box = st.selectbox(
        "Gear Box Type",
        ["Automatic", "Manual", "Tiptronic", "Variator"],
    )

    drive_wheels = st.selectbox(
        "Drive Wheels", ["Front", "Rear", "4x4"]
    )

    wheel = st.selectbox("Wheel", ["Left wheel", "Right-hand drive"])

    color = st.selectbox(
        "Color",
        [
            "Black", "White", "Silver", "Grey", "Blue",
            "Red", "Green", "Brown", "Yellow", "Other",
        ],
    )

# Prediction
if st.button("Predict Price", type="primary", use_container_width=True):

    car_data = {
        "Manufacturer": manufacturer,
        "Model": model,
        "Prod. year": prod_year,
        "Engine volume": engine_volume,
        "Mileage": mileage,
        "Cylinders": cylinders,
        "Levy": levy,
        "Airbags": airbags,
        "Category": category,
        "Leather interior": leather,
        "Fuel type": fuel_type,
        "Gear box type": gear_box,
        "Drive wheels": drive_wheels,
        "Wheel": wheel,
        "Color": color,
    }

    try:
        price = predict_price(car_data)

        st.success(f"Predicted Car Price: {price:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")